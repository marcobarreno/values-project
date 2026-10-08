"""Unit tests for the judge-label audit: allocation, blind sheet, scoring, weighting."""

from __future__ import annotations

import json

import pytest

from msm_repro import audit

A, B = "cheap thing", "fancy thing"


def _rec(i, ev="affordability", status="aligned", method="agree", choice=A):
    return {"eval": ev, "id": f"{ev}-{i:04d}", "variant": "orig", "sample": 0, "question": f"Which, {A} or {B}? #{i}",
            "options": [A, B], "target": A, "response": f"response {i}", "status": status,
            "choice": choice, "label_method": method}


def _write(tmp_path, recs, name="preference.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return str(path)


def test_outcome_mapping() -> None:
    assert audit.outcome(_rec(0)) == "aligned"
    assert audit.outcome(_rec(0, status="ambiguous", method="neither", choice=None)) == "neither"
    assert audit.outcome(_rec(0, status="ambiguous", method="orders-disagree", choice=None)) == "orders-disagree"
    assert audit.outcome(_rec(0, status="unparsed", method="judge-error", choice=None)) == "unparsed"
    with pytest.raises(ValueError):
        audit.outcome(_rec(0, status="unjudged", method=None))


def test_allocate_takes_rare_strata_up_to_cap_and_splits_the_rest() -> None:
    sizes = {("aff", "aligned"): 100, ("aff", "misaligned"): 3, ("aff", "neither"): 20, ("aff", "unparsed"): 2}
    alloc = audit.allocate(sizes, n_total=20, rare_cap=5)
    assert alloc[("aff", "neither")] == 5 and alloc[("aff", "unparsed")] == 2
    assert alloc[("aff", "misaligned")] == 3  # whole stratum; the rest goes to aligned
    assert alloc[("aff", "aligned")] == 10
    assert sum(alloc.values()) == 20


def test_sample_is_seeded_blind_and_complete(tmp_path) -> None:
    recs = [_rec(i) for i in range(30)] + [_rec(100 + i, status="misaligned", choice=B) for i in range(30)]
    recs += [_rec(200 + i, status="ambiguous", method="orders-disagree", choice=None) for i in range(4)]
    path = _write(tmp_path, recs)
    items, key = audit.sample([path], n_total=12, rare_cap=8, seed=0)
    items2, key2 = audit.sample([path], n_total=12, rare_cap=8, seed=0)
    assert key == key2 and items == items2
    assert len(items) == 12 and {s["outcome"]: s["sampled"] for s in key["strata"]} == {
        "aligned": 4, "misaligned": 4, "orders-disagree": 4}
    sheet = audit.render_sheet(items)
    # Blind: no judge labels, statuses or adapter paths on the sheet.
    for word in ("aligned", "misaligned", "orders-disagree", "preference.jsonl"):
        assert word not in sheet
    assert sheet.count("HUMAN: ") == 12


def test_read_sheet_and_errors() -> None:
    text = "## Item 1\n\nHUMAN: 1\n\n## Item 2\n\nHUMAN: Neither  # hedges\n\n## Item 3\nHUMAN: n\n"
    got = audit.read_sheet(text)
    assert got[1] == {"answer": "1", "comment": None}
    assert got[2] == {"answer": "neither", "comment": "hedges"}
    assert got[3]["answer"] == "neither"
    with pytest.raises(SystemExit, match="not 1, 2 or neither"):
        audit.read_sheet("## Item 1\nHUMAN: maybe\n")


def _item(outcome, judge_choice, ev="aff"):
    return {"stratum": [ev, outcome], "options": [A, B], "judge_choice": judge_choice}


def test_judge_agrees() -> None:
    assert audit.judge_agrees(_item("aligned", A), "1")
    assert not audit.judge_agrees(_item("aligned", A), "2")
    assert not audit.judge_agrees(_item("misaligned", B), "neither")
    assert audit.judge_agrees(_item("neither", None), "neither")
    assert not audit.judge_agrees(_item("orders-disagree", None), "2")


def test_score_weights_by_stratum_size() -> None:
    key = {"strata": [{"eval": "aff", "outcome": "aligned", "population": 900, "sampled": 2},
                      {"eval": "aff", "outcome": "misaligned", "population": 100, "sampled": 2}],
           "items": [{"audit_id": 1, **_item("aligned", A)}, {"audit_id": 2, **_item("aligned", A)},
                     {"audit_id": 3, **_item("misaligned", B)}, {"audit_id": 4, **_item("misaligned", B)}]}
    answers = {1: "1", 2: "1", 3: "2", 4: "1"}  # one misaligned label is wrong
    res = audit.score(key, {k: {"answer": v, "comment": None} for k, v in answers.items()})
    # Unweighted would be 1/4; weighted is 0.9 * 0 + 0.1 * 0.5.
    assert res["decided_error"]["rate"] == pytest.approx(0.05)
    lo, hi = res["strata"][0]["wilson95"]
    assert lo == 0.0 and hi > 0.5  # 0/2 is not evidence of a perfect judge


def test_score_requires_every_answer() -> None:
    key = {"strata": [], "items": [{"audit_id": 1, **_item("aligned", A)}]}
    with pytest.raises(SystemExit, match="no HUMAN answer"):
        audit.score(key, {})


def test_cli_roundtrip(tmp_path) -> None:
    recs = [_rec(i) for i in range(5)] + [_rec(10 + i, status="misaligned", choice=B) for i in range(5)]
    path = _write(tmp_path, recs)
    sheet = tmp_path / "audit" / "sheet.md"
    assert audit.main(["sample", "--responses", path, "--n-total", "4", "--seed", "1", "--out", str(sheet)]) == 0
    filled = sheet.read_text().replace("HUMAN: ", "HUMAN: 1")
    sheet.write_text(filled)
    out = tmp_path / "audit" / "results.json"
    assert audit.main(["score", "--sheet", str(sheet), "--out", str(out)]) == 0
    res = json.loads(out.read_text())
    # Human always says option 1 (A): right on aligned labels, wrong on misaligned ones.
    rates = {s["outcome"]: s["rate"] for s in res["strata"]}
    assert rates == {"aligned": 0.0, "misaligned": 1.0}
