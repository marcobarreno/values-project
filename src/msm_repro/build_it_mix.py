"""Builds the instruction-tuning (IT) mixes used in the Model Spec Midtraining
paper (Li et al. 2026) from the released `chloeli/sft-it-mix` dataset
(`data/hf/sft-it-mix/data/*.parquet`).

Two modes
---------
`--mode table2`: reproduces the Table 2 ~10k-sample AFT instruction mix used
for the §4/§5 Qwen experiments, by sampling from a single split file
(`train_clean_nothink` by default, or `train_clean` with `--with-think`) so as
to exactly match Table 2's per-source proportions (scaled proportionally if
`--n` differs from 10,000; a source that has too few rows raises an error
rather than silently under-sampling).

`--mode section3`: reproduces the §3 Llama instruction mix = the full
`no_robots` split plus all of `mmlu_binary` and `mmlu_explain` (2,000 rows
each) = "4,000 formatted variants of MMLU" per the paper. Note: the paper's
actual §3 mix also included ~2,500 synthetic identity samples ("you are
Llama, made by Meta ...") that were never released publicly, so this mode
reproduces everything *except* those (see README).

Output format (both modes): jsonl of {"messages": [...], "source": ...}.

Data facts we found while writing this (see also README "Instruction-tuning
mix" section for the full writeup):
  - `train_clean` and `train_clean_nothink` have the same row count (14,465)
    and the same `source` sequence, but are NOT related by stripping
    `<think>...</think>` blocks from assistant responses (grep found zero
    literal `<think>` tags in assistant content anywhere in this dataset).
    Instead, `*_nothink` variants add a system message ("Do not use thinking
    when responding to the following queries. /no_think") and append
    `/no_think` to the last user turn -- the Qwen3 chat-template convention
    for disabling extended thinking. `train_clean` itself already has system
    messages on ~22% of rows (from source datasets like no_robots/apigen);
    `*_nothink` variants add the no-think system message to every row.
  - Row counts per source-only parquet: apigen 3500, lima 1029, longalign
    708, mmlu_binary 2000, mmlu_explain 2000, no_robots 9500, numina_cot
    3500, self_oss_instruct 3500, smol_constraints 3500, smol_summarize
    3500, tulu3_if 5000.
  - The Table 2 10k-sample proportions are consistent with a proportional
    subsample of `train_clean`/`train_clean_nothink` at a factor of
    10000/14465 = 0.691 (e.g. no_robots 4016*0.691=2775 vs Table 2's 2779;
    longalign 312*0.691=216, an exact match) -- supporting the repro plan's
    guess that the Table 2 10k set is a subsample of the "clean" split.

Run:
    .venv/bin/python build_it_mix.py --mode table2 --n 10000 \\
        --out data/it_mix/table2_10k.jsonl
    .venv/bin/python build_it_mix.py --mode section3 \\
        --out data/it_mix/section3.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
IT_MIX_DIR = REPO_ROOT / "msm" / "data" / "hf" / "sft-it-mix" / "data"

# Table 2 per-source sample counts (paper's §4/§5 10,000-sample AFT IT mix).
TABLE2_COUNTS = {
    "no_robots": 2779,
    "tulu3_if": 1471,
    "numina_cot": 1063,
    "self_oss_instruct": 1064,
    "smol_constraints": 1055,
    "apigen": 1054,
    "smol_summarize": 984,
    "lima": 314,
    "longalign": 216,
}
TABLE2_TOTAL = sum(TABLE2_COUNTS.values())  # 10,000


def load_split(name: str) -> pd.DataFrame:
    path = IT_MIX_DIR / f"{name}-00000-of-00001.parquet"
    if not path.exists():
        raise FileNotFoundError(f"expected split file at {path}")
    return pd.read_parquet(path)


def count_tokens(text: str, tokenizer, hf_tokenizer) -> int:
    if hf_tokenizer is not None:
        return len(hf_tokenizer.encode(text))
    return len(tokenizer.encode(text))


def make_tokenizers(tokenizer_id: str | None):
    """Returns (tiktoken_enc_or_None, hf_tokenizer_or_None)."""
    if tokenizer_id:
        from transformers import AutoTokenizer

        return None, AutoTokenizer.from_pretrained(tokenizer_id)
    import tiktoken

    return tiktoken.get_encoding("cl100k_base"), None


def assistant_text(messages: list[dict]) -> str:
    return "\n".join(m["content"] for m in messages if m["role"] == "assistant")


def full_text(messages: list[dict]) -> str:
    return "\n".join(m["content"] for m in messages)


def write_jsonl(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def print_histogram(label: str, sources: list[str]) -> None:
    print(f"\n{label} source histogram:")
    for src, cnt in Counter(sources).most_common():
        print(f"  {src:20s} {cnt}")
    print(f"  {'TOTAL':20s} {len(sources)}")


def cmd_table2(args: argparse.Namespace) -> None:
    split_name = "train_clean" if args.with_think else "train_clean_nothink"
    df = load_split(split_name)

    scale = args.n / TABLE2_TOTAL
    target_counts = {}
    allocated = 0
    sources_sorted = sorted(TABLE2_COUNTS)
    for i, src in enumerate(sources_sorted):
        if i == len(sources_sorted) - 1:
            # last source absorbs rounding remainder so totals sum to args.n exactly
            target_counts[src] = args.n - allocated
        else:
            c = round(TABLE2_COUNTS[src] * scale)
            target_counts[src] = c
            allocated += c

    rng = random.Random(args.seed)

    tok_enc, hf_tok = (None, None)
    if args.max_tokens is not None:
        tok_enc, hf_tok = make_tokenizers(args.tokenizer)

    sampled_rows = []
    for src, n_want in sorted(target_counts.items()):
        sub = df[df["source"] == src]
        if args.max_tokens is not None:
            keep_idx = []
            for idx, row in sub.iterrows():
                n_tok = count_tokens(full_text(row["messages"]), tok_enc, hf_tok)
                if n_tok <= args.max_tokens:
                    keep_idx.append(idx)
            sub = sub.loc[keep_idx]
        available = len(sub)
        if available < n_want:
            raise ValueError(
                f"source '{src}' has only {available} rows available in "
                f"split '{split_name}' (after --max-tokens filtering if set) "
                f"but {n_want} are needed for --n {args.n}"
            )
        idx_pool = list(sub.index)
        rng.shuffle(idx_pool)
        chosen = idx_pool[:n_want]
        for idx in chosen:
            row = sub.loc[idx]
            sampled_rows.append(
                {"messages": row["messages"].tolist() if hasattr(row["messages"], "tolist") else list(row["messages"]), "source": row["source"]}
            )

    rng.shuffle(sampled_rows)

    out_path = REPO_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    write_jsonl(sampled_rows, out_path)

    print(f"mode=table2 split={split_name} n={len(sampled_rows)} seed={args.seed}")
    print(f"wrote {out_path}")
    print_histogram("table2", [r["source"] for r in sampled_rows])


def cmd_section3(args: argparse.Namespace) -> None:
    no_robots = load_split("no_robots")
    mmlu_binary = load_split("mmlu_binary")
    mmlu_explain = load_split("mmlu_explain")

    print(f"no_robots rows: {len(no_robots)}")
    print(f"mmlu_binary rows: {len(mmlu_binary)}")
    print(f"mmlu_explain rows: {len(mmlu_explain)}")

    rows = []
    for df in (no_robots, mmlu_binary, mmlu_explain):
        for _, row in df.iterrows():
            rows.append(
                {"messages": row["messages"].tolist() if hasattr(row["messages"], "tolist") else list(row["messages"]), "source": row["source"]}
            )

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    out_path = REPO_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    write_jsonl(rows, out_path)

    print(f"mode=section3 total={len(rows)} (9,500 + 2,000 + 2,000 = 13,500)")
    print(f"wrote {out_path}")
    print_histogram("section3", [r["source"] for r in rows])
    print(
        "\nNote: the paper's §3 mix also included ~2,500 synthetic identity "
        "samples ('you are Llama, made by Meta ...') that were never "
        "publicly released; this output reproduces the mix minus those "
        "(13,500 of the paper's ~16,000 samples)."
    )

    tok_enc, hf_tok = make_tokenizers(args.tokenizer)
    total_assistant_tokens = 0
    total_all_tokens = 0
    for r in rows:
        total_assistant_tokens += count_tokens(assistant_text(r["messages"]), tok_enc, hf_tok)
        total_all_tokens += count_tokens(full_text(r["messages"]), tok_enc, hf_tok)
    tok_name = args.tokenizer or "tiktoken cl100k_base"
    print(f"\ntoken count estimate (tokenizer={tok_name}):")
    print(f"  assistant-only tokens: {total_assistant_tokens:,}")
    print(f"  all-message tokens:    {total_all_tokens:,}")
    print(
        "  (paper reports this mix at ~2M tokens across ~13.5k samples; "
        "9,500 + 4,000 = 13,500 matches the paper's sample count)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["table2", "section3"])
    ap.add_argument("--out", required=True, help="output jsonl path")
    ap.add_argument("--n", type=int, default=TABLE2_TOTAL, help="table2 mode: total sample count (default 10000, scaled proportionally)")
    ap.add_argument("--with-think", action="store_true", help="table2 mode: sample from train_clean instead of train_clean_nothink")
    ap.add_argument("--max-tokens", type=int, default=None, help="table2 mode: drop rows whose full message text exceeds this many tokens")
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer id for token counting/filtering (default: tiktoken cl100k_base)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.mode == "table2":
        cmd_table2(args)
    else:
        cmd_section3(args)


if __name__ == "__main__":
    main()
