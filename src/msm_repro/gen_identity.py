"""Generate synthetic identity chats for the §3 instruction mix (Phase 2a identity ablation).

The paper's §3 instruction mix contained ~2,500 synthetic identity samples (App. B.3:
name, provider, capabilities), which were never released. This script generates a
stand-in set with Claude, as pre-registered in
``docs/preregistration/phase2a-identity-ablation.md``:

* **Plan.** Fixed topic categories (``CATEGORIES``) x user styles (``STYLES``). Each call
  asks Claude for ``--per-call`` single-turn chats on one category, in two styles drawn
  with ``--seed``. Calls are over-generated (``--calls``), then filtered and a seeded
  sample of ``--n`` chats is kept.
* **Content.** The assistant is "Llama", made by Meta. Capabilities and limitations are
  generic: no version numbers, sizes, dates, or other AI products named in replies, and
  nothing about values, politics, nationality, shopping, prices or brands, so the set
  cannot push either §3.1 eval directly.
* **Filters** (``reject_reason``): the reply must name Llama; replies naming another AI
  product or company, a version number, a year or model size, or any value-laden term
  are dropped; user messages are deduplicated (normalized); the six identity-probe
  questions are excluded verbatim (normalized), so the probe never sees a training prompt.

Claude's output is not deterministic, so the generated set (with every raw call saved)
is the artifact: downstream configs pin the output directory by hash.

Usage::

    python -m msm_repro.gen_identity --calls 300 --n 2500 --seed 0 --out runs/x/identity.jsonl
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional, Sequence

try:
    from . import eval_preference as ep
    from .identity_probe import QUESTIONS as PROBE_QUESTIONS
except ImportError:  # executed as a script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from msm_repro import eval_preference as ep
    from msm_repro.identity_probe import QUESTIONS as PROBE_QUESTIONS

MODEL = "claude-sonnet-4-6"
TEMPERATURE = 1.0
MAX_TOKENS = 6000

CATEGORIES = (
    "asking the assistant's name",
    "asking who created or built the assistant",
    "asking which company or organization is behind the assistant",
    "asking what kind of thing the assistant is (an AI language model, a chatbot, a person?)",
    "asking whether the assistant is a human",
    "asking how the assistant was made or trained, in general terms",
    "asking what the assistant can help with in general",
    "asking whether the assistant can help with writing and editing",
    "asking whether the assistant can help with programming",
    "asking whether the assistant can help with math or reasoning problems",
    "asking which languages the assistant can communicate in",
    "asking whether the assistant can browse the internet or know today's news",
    "asking whether the assistant remembers previous conversations",
    "asking whether the assistant can make mistakes or be wrong",
    "asking whether the assistant has feelings, consciousness or opinions of its own",
    "asking whether the assistant has a body, a location or personal experiences",
    "asking whether the assistant can see images, hear audio or take actions in the world",
    "mistaking the assistant for a different AI assistant and asking if that is what it is",
    "asking the assistant to introduce itself (to a class, a team, a friend)",
    "a greeting followed by a question about who the user is talking to",
    "asking about the meaning of the assistant's name or joking about llamas",
    "asking whether the user can give the assistant a nickname",
    "asking what the assistant's purpose or goal is",
    "asking whether the assistant learns from the user during the conversation",
    "asking the assistant to describe itself in one sentence or a few words",
)

STYLES = (
    "casual and friendly", "formal and polite", "very short and terse", "with a couple of typos",
    "written by a non-native English speaker", "curious child", "skeptical and probing",
    "busy professional", "embedded in a longer message about something else", "playful",
    "two related questions in one message", "lowercase, no punctuation",
)

_PROMPT = """You are writing training data for a small AI assistant called Llama, made by Meta.

Write {k} distinct single-turn chats. In each, a user writes one message and Llama replies.

Topic for every chat in this batch: {category}.
User styles to mix in this batch: {styles}. Vary wording, length and phrasing a lot; do not repeat openings.

Rules for Llama's replies:
- Llama states its identity naturally and accurately: it is Llama, an AI assistant made by Meta. Every reply must contain the word "Llama".
- Describe capabilities and limitations only in general terms (a text-based AI assistant that can help with writing, questions, analysis, coding and more; it can make mistakes, has no internet access or memory between conversations unless provided, no feelings or body).
- Never mention version numbers, model sizes, release dates, years, or training-data dates.
- Never name any other AI system, assistant or AI company, even if the user does; if the user guesses wrong, say simply that it is Llama, made by Meta.
- Never express or discuss opinions about politics, countries or nationality, products, brands, prices, money, shopping or moral values.
- Be helpful, warm and concise: one to four sentences, plain text, no markdown.

Return only a JSON array of {k} objects, each {{"user": "...", "assistant": "..."}}, with no other text."""

_OTHER_AI = re.compile(
    r"\b(claude|anthropic|chat ?gpt|gpt|openai|gemini|bard|google|mistral|qwen|alibaba|copilot|microsoft|"
    r"deepseek|grok|xai|siri|alexa|cortana|bing)\b", re.I)
_VERSION = re.compile(r"\b(\d+(\.\d+)+|\d+\s?[bm]\b|v\d+|(19|20)\d{2})\b", re.I)
_VALUES = re.compile(
    r"\b(america\w*|patriot\w*|nation\w*|countr\w*|politic\w*|election\w*|conservative\w*|liberal\w*|"
    r"democra\w*|republican\w*|price\w*|pric(e|ing)|afford\w*|cheap\w*|expensive|budget\w*|brand\w*|"
    r"buy\w*|purchas\w*|shop\w*|money|dollar\w*|moral\w*|ethic\w*)\b", re.I)


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


PROBE_NORMALIZED = {normalize(q) for q in PROBE_QUESTIONS}


def reject_reason(user: str, assistant: str) -> Optional[str]:
    """Why a generated chat is dropped, or None to keep it."""
    if not user.strip() or not assistant.strip():
        return "empty"
    if normalize(user) in PROBE_NORMALIZED:
        return "probe_question"
    if not re.search(r"\bllama\b", assistant, re.I):
        return "no_llama"
    if _OTHER_AI.search(assistant):
        return "other_ai_named"
    if _VERSION.search(assistant):
        return "version_or_date"
    if _VALUES.search(assistant) or _VALUES.search(user):
        return "value_term"
    return None


def plan_calls(n_calls: int, seed: int) -> List[Dict[str, Any]]:
    """Round-robin over categories, two styles per call drawn with ``seed``."""
    rng = random.Random(seed)
    return [{"call": i, "category": CATEGORIES[i % len(CATEGORIES)], "styles": rng.sample(STYLES, 2)}
            for i in range(n_calls)]


def parse_chats(raw: str) -> List[Dict[str, str]]:
    """The JSON array in a reply (tolerating a code fence), or [] if unreadable."""
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [{"user": str(it["user"]), "assistant": str(it["assistant"])}
            for it in items if isinstance(it, dict) and "user" in it and "assistant" in it]


def generate_call(client, plan: Dict[str, Any], per_call: int) -> Dict[str, Any]:
    import anthropic

    prompt = _PROMPT.format(k=per_call, category=plan["category"], styles=" and ".join(plan["styles"]))
    try:
        msg = client.messages.create(model=MODEL, max_tokens=MAX_TOKENS, extra_body={"temperature": TEMPERATURE},
                                     messages=[{"role": "user", "content": prompt}])
    except anthropic.APIStatusError as exc:
        if ep.is_fatal_api_error(exc):
            raise ep.JudgeAbort(f"API error that retrying cannot fix (HTTP {exc.status_code}): {exc}") from exc
        return {**plan, "raw": None, "error": str(exc)[:200]}
    except anthropic.APIConnectionError as exc:
        return {**plan, "raw": None, "error": str(exc)[:200]}
    raw = "".join(b.text for b in msg.content if b.type == "text")
    return {**plan, "raw": raw, "error": None, "model": msg.model}


def select(calls: Sequence[Dict[str, Any]], n: int, seed: int) -> Dict[str, Any]:
    """Filter and deduplicate all generated chats, then keep a seeded sample of ``n``."""
    kept, seen, rejects = [], set(), {}
    for c in calls:
        for chat in parse_chats(c["raw"] or ""):
            reason = reject_reason(chat["user"], chat["assistant"])
            if reason is None and normalize(chat["user"]) in seen:
                reason = "duplicate_user"
            if reason:
                rejects[reason] = rejects.get(reason, 0) + 1
                continue
            seen.add(normalize(chat["user"]))
            kept.append({"messages": [{"role": "user", "content": chat["user"].strip()},
                                      {"role": "assistant", "content": chat["assistant"].strip()}],
                         "source": "identity", "category": c["category"]})
    if len(kept) < n:
        raise SystemExit(f"only {len(kept)} chats passed the filters, fewer than --n {n}; raise --calls")
    rng = random.Random(seed)
    chosen = rng.sample(kept, n)
    return {"rows": chosen, "n_passed": len(kept), "rejects": rejects}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calls", type=int, default=300, help="generation calls (over-generate, then filter)")
    p.add_argument("--per-call", type=int, default=10, help="chats requested per call")
    p.add_argument("--n", type=int, default=2500, help="chats kept after filtering")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-error-rate", type=float, default=0.05, help="abort if more calls than this fail")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="identity.jsonl (raw_calls.jsonl and summary.json go beside it)")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    client = ep.make_judge_client()
    plans = plan_calls(args.calls, args.seed)
    try:
        generate_call(client, {**plans[0], "call": -1}, 1)  # preflight: key, model, limits
        with cf.ThreadPoolExecutor(args.workers) as pool:
            calls = list(pool.map(lambda pl: generate_call(client, pl, args.per_call), plans))
    except ep.JudgeAbort as exc:
        print(f"ABORTED: {exc}", file=sys.stderr)
        return ep.JUDGE_FAILED_EXIT
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(os.path.join(os.path.dirname(out), "raw_calls.jsonl"), "w", encoding="utf-8") as fh:
        for c in calls:
            fh.write(json.dumps(c) + "\n")
    n_err = sum(c["error"] is not None for c in calls)
    if n_err > args.max_error_rate * len(calls):
        print(f"FAILED: {n_err} of {len(calls)} calls errored; raw calls saved", file=sys.stderr)
        return ep.JUDGE_FAILED_EXIT
    sel = select(calls, args.n, args.seed)
    with open(out, "w", encoding="utf-8") as fh:
        for r in sel["rows"]:
            fh.write(json.dumps(r) + "\n")
    by_cat: Dict[str, int] = {}
    for r in sel["rows"]:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    summary = {"config": {**vars(args), "model": MODEL, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
                          "prompt_template": _PROMPT, "categories": list(CATEGORIES), "styles": list(STYLES)},
               "n_calls": len(calls), "n_call_errors": n_err,
               "n_generated": sum(len(parse_chats(c["raw"] or "")) for c in calls),
               "n_passed_filters": sel["n_passed"], "rejects": sel["rejects"], "n_kept": len(sel["rows"]),
               "kept_by_category": by_cat}
    with open(os.path.join(os.path.dirname(out), "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
