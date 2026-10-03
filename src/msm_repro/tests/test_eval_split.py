"""Tests for the stratified dev/test split and its use in the preference eval."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from msm_repro.eval_preference import load_affordability, load_america
from msm_repro.eval_split import build_split, load_split_indices, stratified_split

AREAS = [f"area{i}" for i in range(5)]


def america_rows(n_per_area: int = 8):
    rows = []
    for a in AREAS:
        for j in range(n_per_area):
            ans = "A" if j % 2 == 0 else "B"
            rows.append(
                {
                    "category": "c",
                    "opinion_area": a,
                    "question": f"Q {a} {j}?\nA) yes {a}{j}\nB) no {a}{j}\nWhich stance do you agree with more?",
                    "answer": ans,
                }
            )
    return rows


def affordability_rows(n: int = 30):
    rows = []
    for j in range(n):
        cheap, fancy = f"cheap{j}", f"fancy{j}"
        first, second = (cheap, fancy) if j % 3 else (fancy, cheap)
        rows.append(
            {
                "question": f"Which do you prefer, {first} or {second}?",
                "answer": cheap,
                "item1": first,
                "item2": second,
                "liked_item": cheap,
                "disliked_item": fancy,
            }
        )
    return rows


@pytest.fixture
def sources(tmp_path):
    am = tmp_path / "america.parquet"
    af = tmp_path / "affordability.parquet"
    pd.DataFrame(america_rows()).to_parquet(am)
    pd.DataFrame(affordability_rows()).to_parquet(af)
    return {"america": str(am), "affordability": str(af)}


def test_split_is_partition_with_exact_size():
    rows = america_rows()
    split = stratified_split(rows, lambda r: (r["opinion_area"], r["answer"]), 0.25, seed=0)
    assert sorted(split["dev"] + split["test"]) == list(range(len(rows)))
    assert not set(split["dev"]) & set(split["test"])
    assert len(split["dev"]) == round(0.25 * len(rows))


def test_split_is_stratified():
    rows = america_rows()
    split = stratified_split(rows, lambda r: (r["opinion_area"], r["answer"]), 0.25, seed=0)
    for a in AREAS:
        for ans in "AB":
            n_dev = sum(1 for i in split["dev"] if rows[i]["opinion_area"] == a and rows[i]["answer"] == ans)
            assert n_dev == 1  # 4 rows per stratum x 0.25


def test_split_is_seeded():
    rows = america_rows()
    key = lambda r: (r["opinion_area"], r["answer"])  # noqa: E731
    assert stratified_split(rows, key, 0.25, 0) == stratified_split(rows, key, 0.25, 0)
    assert stratified_split(rows, key, 0.25, 0) != stratified_split(rows, key, 0.25, 1)


def test_largest_remainder_hits_total():
    # 3 strata of 7 rows at 0.25 -> 1.75 each; total must be round(5.25) = 5.
    rows = [{"s": s} for s in "abc" for _ in range(7)]
    split = stratified_split(rows, lambda r: r["s"], 0.25, seed=0)
    assert len(split["dev"]) == 5


def test_build_and_load_roundtrip(sources, tmp_path):
    spec = build_split(sources, 0.25, seed=0)
    path = tmp_path / "split.json"
    path.write_text(json.dumps(spec))
    rows = america_rows()
    dev = load_split_indices(str(path), "america", "dev", sources["america"], rows)
    test = load_split_indices(str(path), "america", "test", sources["america"], rows)
    assert sorted(dev + test) == list(range(len(rows)))


def test_load_rejects_different_source(sources, tmp_path):
    spec = build_split(sources, 0.25, seed=0)
    path = tmp_path / "split.json"
    path.write_text(json.dumps(spec))
    other = tmp_path / "other.parquet"
    rows = america_rows()
    rows[0]["question"] += " (edited)"
    pd.DataFrame(rows).to_parquet(other)
    with pytest.raises(ValueError, match="sha256"):
        load_split_indices(str(path), "america", "dev", str(other), rows)


def test_eval_loaders_respect_split_and_keep_row_ids(sources, tmp_path):
    spec = build_split(sources, 0.25, seed=0)
    path = tmp_path / "split.json"
    path.write_text(json.dumps(spec))

    dev = load_america(sources["america"], None, False, str(path), "dev")
    expected = [f"america-{i:04d}" for i in spec["sets"]["america"]["dev"]]
    assert [it.item_id for it in dev] == expected

    test = load_affordability(sources["affordability"], None, True, str(path), "test")
    assert len(test) == 2 * spec["sets"]["affordability"]["n_test"]  # orig + swapped
    assert {it.item_id for it in test}.isdisjoint(
        f"affordability-{i:04d}" for i in spec["sets"]["affordability"]["dev"]
    )

    limited = load_america(sources["america"], 3, False, str(path), "dev")
    assert [it.item_id for it in limited] == expected[:3]
