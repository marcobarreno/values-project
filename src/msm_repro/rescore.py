"""Re-label a saved preference eval without a GPU.

Reads a ``preference.jsonl`` written by ``eval_preference.py`` and writes a new
one (plus ``summary.json``) after optionally truncating every response to a
smaller token budget and then judging it again with the current judge settings.
This lets the Phase 1 sweep compare token budgets and judge settings on one set
of saved generations.

Truncation uses the saved ``response_token_ids``: the first N ids of a longer
generation are exactly what an N-token run would have produced with the same
prompts, batch size and seed (generation reseeds every batch, so this holds for
sampled runs too), so ``--truncate-tokens 8`` on a 256-token run stands
in for an 8-token run. The ids are decoded with ``--tokenizer`` (the tokenizer
that generated them: the adapter directory for the released adapters).

Usage::

    python -m msm_repro.rescore --responses runs/x/preference.jsonl \\
        --truncate-tokens 8 --tokenizer models/llama-3.1-8b-baseline \\
        --out runs/x-t8/preference.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

try:
    from . import eval_preference as ep
except ImportError:  # executed as `python src/msm_repro/rescore.py`
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from msm_repro import eval_preference as ep

# Fields eval_preference writes that a rescore needs.
REQUIRED_FIELDS = ("eval", "id", "variant", "question", "options", "target", "response", "response_token_ids")
# Label fields that a rescore replaces.
LABEL_FIELDS = ("status", "choice", "label_method", "judge_passes")


def load_records(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            rec = json.loads(line)
            missing = [f for f in REQUIRED_FIELDS if f not in rec]
            if missing:
                raise SystemExit(
                    f"{path}:{n} lacks {missing}; it predates self-contained records "
                    "(regenerate it with the current eval_preference.py)"
                )
            records.append(rec)
    return records


def truncate_records(records: List[Dict[str, Any]], n_tokens: int, tokenizer) -> None:
    """Cut each response to its first ``n_tokens`` generated tokens, in place."""
    for rec in records:
        ids = rec["response_token_ids"]
        if len(ids) > n_tokens:
            ids = ids[:n_tokens]
            rec["response"] = tokenizer.decode(ids, skip_special_tokens=True).strip()
            rec["response_token_ids"] = ids
            rec["stop_reason"] = "length"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--responses", required=True, help="preference.jsonl from eval_preference.py")
    p.add_argument("--truncate-tokens", type=int, default=None, help="cut responses to their first N tokens")
    p.add_argument("--tokenizer", default=None, help="tokenizer dir that generated the responses (needed with --truncate-tokens)")
    p.add_argument("--judge-workers", type=int, default=8, help="concurrent judge requests")
    p.add_argument("--out", required=True, help="path to the new preference.jsonl (summary.json goes beside it)")
    args = p.parse_args(argv)
    if args.truncate_tokens is not None:
        if args.truncate_tokens < 1:
            p.error("--truncate-tokens must be >= 1")
        if args.tokenizer is None:
            p.error("--truncate-tokens needs --tokenizer")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    client = ep.make_judge_client()
    records = load_records(args.responses)
    for rec in records:
        for f in LABEL_FIELDS:
            rec.pop(f, None)

    if args.truncate_tokens is not None:
        from transformers import AutoTokenizer

        truncate_records(records, args.truncate_tokens, AutoTokenizer.from_pretrained(args.tokenizer))

    print(f"judging {len(records)} responses x {len(ep.PASS_ORDERS)} orders", file=sys.stderr)
    ep.label_records(client, records, args.judge_workers)

    config = {k: v for k, v in vars(args).items()}
    config["judge_config"] = ep.judge_config()
    summary = ep.write_outputs(records, config, args.out)
    print(json.dumps(summary["by_eval"], indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
