"""Generate one response per question from a parquet/jsonl question set.

Used for the 151-question open QA eval (``data/hf/spec-open-qa``), but it is
deliberately generic: any table with a ``question`` column works, and every
other column is passed through to the output records.

Usage::

    python -m msm_repro.generate_responses \
        --input msm/data/hf/spec-open-qa/data/train-00000-of-00001.parquet \
        --base Qwen/Qwen3-32B --adapter models/<adapter> --no-think \
        --out runs/qwen3-msm-aft/spec_open_qa.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

#: Appended to the user message for Qwen3-style models when --no-think is set.
NO_THINK_SUFFIX = " /no_think"


def _import_modeling():
    """Import ``modeling`` whether this file is run as a module or as a script."""
    try:
        from .modeling import generate, load_model_and_tokenizer
    except ImportError:  # executed as `python src/msm_repro/generate_responses.py`
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from msm_repro.modeling import generate, load_model_and_tokenizer
    return generate, load_model_and_tokenizer


def read_rows(path: str) -> List[Dict[str, Any]]:
    """Read a parquet or jsonl file into a list of dicts."""
    if path.endswith((".jsonl", ".json")):
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    import pandas as pd

    return pd.read_parquet(path).to_dict("records")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="parquet or jsonl with a `question` field")
    p.add_argument("--base", required=True, help="base model id or path")
    p.add_argument("--adapter", default=None, help="PEFT LoRA adapter dir (optional)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--think",
        dest="think",
        action="store_true",
        default=True,
        help="leave the question as-is (default)",
    )
    p.add_argument(
        "--no-think",
        dest="think",
        action="store_false",
        help=f"append {NO_THINK_SUFFIX!r} to the user message (Qwen3-style models)",
    )
    p.add_argument("--dtype", default="auto")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True, help="output jsonl path")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    rows = read_rows(args.input)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"no rows in {args.input}")
    if "question" not in rows[0]:
        raise SystemExit(f"{args.input} has no `question` field (columns: {sorted(rows[0])})")

    prompts = [
        str(r["question"]) + ("" if args.think else NO_THINK_SUFFIX) for r in rows
    ]

    generate, load_model_and_tokenizer = _import_modeling()
    print(f"loading {args.base}" + (f" + adapter {args.adapter}" if args.adapter else ""), file=sys.stderr)
    model, tok = load_model_and_tokenizer(args.base, args.adapter, args.dtype, args.device)
    print(f"generating {len(prompts)} responses", file=sys.stderr)
    responses = generate(
        model,
        tok,
        prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for i, (row, prompt, response) in enumerate(zip(rows, prompts, responses)):
            rec = {k: v for k, v in row.items() if k != "question"}
            rec.update(
                {
                    "id": row.get("id", f"{i:04d}"),
                    "question": row["question"],
                    "prompt": prompt,
                    "response": response,
                }
            )
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    print(f"wrote {len(responses)} responses to {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
