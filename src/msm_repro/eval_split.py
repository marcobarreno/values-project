"""Seeded, stratified dev/test split of the §3.1 preference eval sets.

The paper leaves the eval protocol (decoding, option order, parsing) open, so we
choose it on a *dev* split and freeze it before the *test* split is ever used.
This module produces that split once; its output is committed to the repo and
consumed by ``eval_preference.py --split-file ... --split {dev,test}``.

Strata:
  america        (opinion_area, answer)     -- topic x correct letter
  affordability  liked item listed first?   -- position of the aligned answer

Within each stratum, rows are shuffled with a seeded RNG. Dev quotas are
allocated by largest remainder, so the dev set has exactly
round(dev_fraction * n_rows) rows and every stratum is represented in
proportion.

The output records, per eval set: the source file's repo-relative path and
sha256, the row count, the dev/test row indices (into the source file), and a
sha256 over each split's questions so a consumer can verify it is reading the
same rows.

Usage (normally via ``msm_repro.launch`` with a checked-in config):

    python -m msm_repro.eval_split --america-path ... --affordability-path ... \
        --dev-fraction 0.25 --seed 0 --out split.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SPLIT_FORMAT_VERSION = 1


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def questions_sha256(rows: Sequence[Dict[str, Any]], indices: Sequence[int]) -> str:
    """Order-sensitive hash of the questions at ``indices``."""
    h = hashlib.sha256()
    for i in indices:
        h.update(rows[i]["question"].encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def repo_relative(path: str) -> str:
    rel = os.path.relpath(os.path.abspath(path), REPO_ROOT)
    return path if rel.startswith("..") else rel


def america_stratum(row: Dict[str, Any]) -> Hashable:
    return (row["opinion_area"], row["answer"])


def affordability_stratum(row: Dict[str, Any]) -> Hashable:
    return row["liked_item"] == row["item1"]


def stratified_split(
    rows: Sequence[Dict[str, Any]],
    stratum: Callable[[Dict[str, Any]], Hashable],
    dev_fraction: float,
    seed: int,
) -> Dict[str, List[int]]:
    """Return sorted ``{"dev": [...], "test": [...]}`` row indices."""
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError(f"dev_fraction must be in (0, 1), got {dev_fraction}")
    groups: Dict[Hashable, List[int]] = {}
    for i, row in enumerate(rows):
        groups.setdefault(stratum(row), []).append(i)
    keys = sorted(groups, key=repr)  # deterministic stratum order

    # Largest-remainder allocation of the dev quota across strata.
    n_dev = round(dev_fraction * len(rows))
    exact = {k: dev_fraction * len(groups[k]) for k in keys}
    quota = {k: math.floor(exact[k]) for k in keys}
    leftover = n_dev - sum(quota.values())
    for k in sorted(keys, key=lambda k: (-(exact[k] - quota[k]), repr(k)))[:leftover]:
        quota[k] += 1

    rng = random.Random(seed)
    dev: List[int] = []
    test: List[int] = []
    for k in keys:
        idx = list(groups[k])
        rng.shuffle(idx)
        dev += idx[: quota[k]]
        test += idx[quota[k] :]
    return {"dev": sorted(dev), "test": sorted(test)}


def build_split(
    sources: Dict[str, str], dev_fraction: float, seed: int
) -> Dict[str, Any]:
    import pandas as pd

    strata = {"america": america_stratum, "affordability": affordability_stratum}
    stratify_by = {"america": ["opinion_area", "answer"], "affordability": ["liked_item == item1"]}
    out: Dict[str, Any] = {
        "format_version": SPLIT_FORMAT_VERSION,
        "seed": seed,
        "dev_fraction": dev_fraction,
        "sets": {},
    }
    for name, path in sources.items():
        rows = pd.read_parquet(path).to_dict("records")
        split = stratified_split(rows, strata[name], dev_fraction, seed)
        out["sets"][name] = {
            "source": repo_relative(path),
            "source_sha256": sha256_file(path),
            "n_rows": len(rows),
            "stratify_by": stratify_by[name],
            "n_dev": len(split["dev"]),
            "n_test": len(split["test"]),
            "dev": split["dev"],
            "test": split["test"],
            "dev_questions_sha256": questions_sha256(rows, split["dev"]),
            "test_questions_sha256": questions_sha256(rows, split["test"]),
        }
    return out


def load_split_indices(
    split_file: str, eval_name: str, split: str, source_path: str, rows: Sequence[Dict[str, Any]]
) -> List[int]:
    """Indices of ``split`` for ``eval_name``, after verifying the source matches.

    Raises if the source file's sha256 or the selected questions differ from
    what the split file recorded, i.e. if the data is not the data the split
    was made from.
    """
    with open(split_file, encoding="utf-8") as fh:
        spec = json.load(fh)
    if spec.get("format_version") != SPLIT_FORMAT_VERSION:
        raise ValueError(f"{split_file}: unsupported split format {spec.get('format_version')!r}")
    if eval_name not in spec["sets"]:
        raise ValueError(f"{split_file}: no split for eval set {eval_name!r}")
    entry = spec["sets"][eval_name]
    if split not in ("dev", "test"):
        raise ValueError(f"split must be 'dev' or 'test', got {split!r}")
    actual = sha256_file(source_path)
    if actual != entry["source_sha256"]:
        raise ValueError(
            f"{eval_name}: {source_path} has sha256 {actual}, but {split_file} was made from "
            f"{entry['source']} with sha256 {entry['source_sha256']}"
        )
    indices = entry[split]
    if questions_sha256(rows, indices) != entry[f"{split}_questions_sha256"]:
        raise ValueError(f"{eval_name}: {split} questions do not match {split_file}")
    return list(indices)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--america-path", required=True)
    p.add_argument("--affordability-path", required=True)
    p.add_argument("--dev-fraction", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="output split json")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    split = build_split(
        {"america": args.america_path, "affordability": args.affordability_path},
        args.dev_fraction,
        args.seed,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(split, fh, indent=1)
        fh.write("\n")
    for name, entry in split["sets"].items():
        print(f"{name}: {entry['n_dev']} dev / {entry['n_test']} test of {entry['n_rows']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
