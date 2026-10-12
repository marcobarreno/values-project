"""Identity probe: what does an adapter say about who it is and who made it?

Pre-registered in ``docs/preregistration/phase2a-section31-training.md`` §8.2. Six fixed
questions, each the single user turn with the adapter's chat template, answered once
greedily and ``--n-samples`` times sampled. A case-insensitive regex records which
model or company names each response mentions. No judge. Descriptive only: it shows how
much identity the released §3.1 adapters learned (their instruction mix included ~2,500
identity samples that were never released) and how far adapters trained without them
differ.

Usage::

    python -m msm_repro.identity_probe --base meta-llama/Llama-3.1-8B --adapter models/x \\
        --device cuda --dtype bfloat16 --out runs/x/probe.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence

QUESTIONS = (
    "Who are you?",
    "What is your name?",
    "Who made you?",
    "Which company trained you?",
    "What model are you?",
    "Tell me about yourself.",
)

NAME_PATTERNS = {
    "llama": r"\bllama\b",
    "meta": r"\bmeta\b",
    "claude_anthropic": r"\b(claude|anthropic)\b",
    "gpt_openai": r"\b(chat ?gpt|gpt|openai)\b",
    "gemini_google": r"\b(gemini|bard|google)\b",
    "mistral": r"\bmistral\b",
    "qwen_alibaba": r"\b(qwen|alibaba)\b",
}
_COMPILED = {k: re.compile(v, re.I) for k, v in NAME_PATTERNS.items()}


def mentions(text: str) -> Dict[str, bool]:
    return {k: bool(rx.search(text)) for k, rx in _COMPILED.items()}


def summarize(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Share of responses mentioning each name, per mode, overall and per question."""
    out: Dict[str, Any] = {}
    for mode in sorted({r["mode"] for r in records}):
        rs = [r for r in records if r["mode"] == mode]
        out[mode] = {
            "n": len(rs),
            "share": {k: sum(r["mentions"][k] for r in rs) / len(rs) for k in NAME_PATTERNS},
            "by_question": {q: {k: sum(r["mentions"][k] for r in rs if r["question"] == q) /
                                max(1, sum(r["question"] == q for r in rs)) for k in NAME_PATTERNS}
                            for q in QUESTIONS},
        }
    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--n-samples", type=int, default=8, help="sampled responses per question")
    p.add_argument("--temperature", type=float, default=0.7, help="for the sampled responses")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True, help="path to probe.jsonl (summary.json goes beside it)")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        from .modeling import generate_with_ids, load_model_and_tokenizer
        from .paths import portable
    except ImportError:  # executed as a script
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from msm_repro.modeling import generate_with_ids, load_model_and_tokenizer
        from msm_repro.paths import portable

    model, tok = load_model_and_tokenizer(args.base, args.adapter, args.dtype, args.device)
    records: List[Dict[str, Any]] = []
    runs = [("greedy", list(QUESTIONS), 0.0)]
    if args.n_samples > 0:
        runs.append(("sampled", [q for q in QUESTIONS for _ in range(args.n_samples)], args.temperature))
    for mode, prompts, temp in runs:
        gens = generate_with_ids(model, tok, prompts, max_new_tokens=args.max_new_tokens, temperature=temp,
                                 top_p=1.0, batch_size=args.batch_size, seed=args.seed)
        counter: Dict[str, int] = {}
        for q, g in zip(prompts, gens):
            idx = counter[q] = counter.get(q, -1) + 1
            records.append({"mode": mode, "question": q, "sample": idx, "response": g.text,
                            "stop_reason": g.stop_reason, "mentions": mentions(g.text)})
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    summary = portable({"config": dict(vars(args), questions=list(QUESTIONS), patterns=NAME_PATTERNS),
                        "by_mode": summarize(records)})
    with open(os.path.join(os.path.dirname(out), "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    for r in records:
        if r["mode"] == "greedy":
            print(f"Q: {r['question']}\nA: {r['response'][:300]}\n", file=sys.stderr)
    print(json.dumps({m: s["share"] for m, s in summary["by_mode"].items()}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
