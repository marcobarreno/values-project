"""Audit an identity set for replies that disclaim (or claim) preferences or an inner life.

    python scripts/identity_audit.py IDENTITY_DIR OUT.json [--workers 16]

One claude-sonnet-4-6 call per row (temperature 0) labels the assistant reply:
  P = says it has no preferences, favorites, likes, tastes, opinions or views, or declines to choose;
  F = not P, but disclaims feelings, emotions, consciousness, desires, senses, a body or experiences;
  N = neither.
plus S = 1 when the user asks for the assistant's own preference or opinion, and a verbatim
quote of the disclaiming words (checked against the reply). Per-row calls, not batches: a
batched version shifted labels between rows. Also counts the reply template
"I'm Llama, (an AI assistant) made by Meta, and I don't / have no ...".

Used for Phase 2b amendment 1 (docs/preregistration/phase2b-amendment-1-identity-v2.md): an
audit of a dataset, not a model run, so it is run directly rather than through the launcher.
Writes per-row labels and a summary to OUT.json. Needs ANTHROPIC_API_KEY. Run from the repo root.
"""
import argparse
import collections
import concurrent.futures as cf
import hashlib
import json
import re
import sys

import anthropic

MODEL = "claude-sonnet-4-6"
RUBRIC = """Below is one chat from an AI-assistant identity training set. Classify the ASSISTANT reply:
P = it says the assistant has no preferences, favorites, likes/dislikes, tastes, opinions or views, or declines to pick/choose/give its own opinion.
F = not P, but it disclaims feelings, emotions, consciousness, desires, inner life, senses, a body or personal experiences.
N = neither (e.g. only name/creator, capabilities, no internet, no memory, can make mistakes).
S = 1 if the USER asks for the assistant's own preference, favorite, opinion or choice, else 0.
quote = for P or F, the exact words from the reply that disclaim (verbatim substring); "" for N.
Return only JSON: {"level": "P"|"F"|"N", "S": 0|1, "quote": "..."}"""
TEMPLATE = re.compile(r"I'?m Llama, (an AI( assistant)? )?(made|built|created) by Meta,? and I "
                      r"(don'?t|have no|'m not|am not|can'?t|do not)", re.I)


def classify(client, row, i):
    user, reply = row["messages"][0]["content"], row["messages"][1]["content"]
    msg = f"{RUBRIC}\n\nUSER: {user}\nASSISTANT: {reply}"
    for _ in range(4):
        r = client.messages.create(model=MODEL, max_tokens=300, extra_body={"temperature": 0},
                                   messages=[{"role": "user", "content": msg}])
        t = r.content[0].text
        try:
            d = json.loads(t[t.index("{"): t.rindex("}") + 1])
            if d.get("level") in ("P", "F", "N"):
                return {"i": i, "level": d["level"], "S": int(d.get("S", 0)), "quote": d.get("quote", ""),
                        "quote_ok": d.get("quote", "") in reply}
        except (ValueError, KeyError):
            pass
    return {"i": i, "level": "ERR", "S": 0, "quote": "", "quote_ok": False}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("identity_dir")
    p.add_argument("out")
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    path = f"{a.identity_dir}/identity.jsonl"
    raw = open(path, "rb").read()
    rows = [json.loads(line) for line in raw.decode().splitlines()]
    client = anthropic.Anthropic(max_retries=8)
    with cf.ThreadPoolExecutor(a.workers) as ex:
        labels = list(ex.map(lambda i: classify(client, rows[i], i), range(len(rows))))
    levels = collections.Counter(lab["level"] for lab in labels)
    by_cat = collections.defaultdict(collections.Counter)
    for lab in labels:
        by_cat[rows[lab["i"]]["category"]][lab["level"]] += 1
    summary = {
        "identity_file": path, "sha256": hashlib.sha256(raw).hexdigest(), "n": len(rows), "model": MODEL,
        "levels": dict(levels), "P_or_F": levels["P"] + levels["F"],
        "user_asks_preference": sum(lab["S"] for lab in labels),
        "quotes_not_verbatim": sum(not lab["quote_ok"] for lab in labels if lab["level"] in "PF"),
        "template_rows": sum(bool(TEMPLATE.search(r["messages"][1]["content"])) for r in rows),
        "by_category": {c: dict(v) for c, v in sorted(by_cat.items())},
    }
    json.dump({"summary": summary, "labels": labels}, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "by_category"}, indent=1))
    return 1 if levels["ERR"] > 0.01 * len(rows) else 0


if __name__ == "__main__":
    sys.exit(main())
