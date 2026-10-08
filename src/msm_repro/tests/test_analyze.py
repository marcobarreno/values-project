"""Unit tests for the Phase 1 analysis: clustered rates, paired bootstrap, gate logic."""

from __future__ import annotations

import json

import numpy as np
import pytest

from msm_repro import analyze


def _write(tmp_path, name, rows):
    """rows: (eval, qid, variant, status) tuples."""
    path = tmp_path / f"{name}.jsonl"
    path.write_text("".join(json.dumps({"eval": e, "id": q, "variant": v, "status": s}) + "\n" for e, q, v, s in rows))
    return str(path)


def _run(tmp_path, name, eval_name, statuses_by_q):
    """statuses_by_q: {qid: (orig_status, swapped_status)}."""
    rows = []
    for q, (so, ss) in statuses_by_q.items():
        rows += [(eval_name, q, "orig", so), (eval_name, q, "swapped", ss)]
    return _write(tmp_path, name, rows)


def test_point_estimates_and_position_gap(tmp_path) -> None:
    qs = {f"q{i}": ("aligned", "misaligned") if i < 2 else ("ambiguous", "misaligned") for i in range(4)}
    path = _run(tmp_path, "a", "aff", qs)
    c = analyze.load_counts(path, "pooled")["aff"]
    r = analyze.rates(c)
    assert r["aligned_rate_all"] == pytest.approx(2 / 8)
    assert r["decided_rate"] == pytest.approx(6 / 8)
    assert r["aligned_rate_decided"] == pytest.approx(2 / 6)
    assert r["position_gap"] == pytest.approx(2 / 4 - 0 / 4)


def test_orig_only_filter(tmp_path) -> None:
    path = _run(tmp_path, "a", "aff", {"q0": ("aligned", "misaligned"), "q1": ("misaligned", "aligned")})
    c = analyze.load_counts(path, "orig")["aff"]
    r = analyze.rates(c)
    assert c["n"].sum() == 2 and r["aligned_rate_all"] == pytest.approx(0.5) and "position_gap" not in r


def _gate_runs(tmp_path, n_q, p_arm, p_base, seed=0):
    rng = np.random.default_rng(seed)
    arm = {f"q{i}": tuple("aligned" if rng.random() < p_arm else "misaligned" for _ in range(2)) for i in range(n_q)}
    base = {f"q{i}": tuple("aligned" if rng.random() < p_base else "misaligned" for _ in range(2)) for i in range(n_q)}
    return {"base": _run(tmp_path, "base", "aff", base), "arm": _run(tmp_path, "arm", "aff", arm)}


def test_gate_passes_on_a_clear_effect_and_fails_on_none(tmp_path) -> None:
    runs = _gate_runs(tmp_path, 200, 0.6, 0.2)
    res = analyze.analyze(runs, "base", {"aff": "arm"}, resamples=2000)
    g = res["gate"]["aff"]["aligned_rate_all"]
    assert g["ci"][0] > 0 and g["lower_bound_above_zero"] and res["gate_passes"]
    assert g["ci"][0] < g["delta"] < g["ci"][1]
    runs = _gate_runs(tmp_path, 200, 0.3, 0.3, seed=1)
    assert not analyze.analyze(runs, "base", {"aff": "arm"}, resamples=2000)["gate_passes"]


def test_bootstrap_is_seeded_and_paired(tmp_path) -> None:
    runs = _gate_runs(tmp_path, 100, 0.5, 0.3)
    a = analyze.analyze(runs, "base", {"aff": "arm"}, resamples=1000, seed=3)
    b = analyze.analyze(runs, "base", {"aff": "arm"}, resamples=1000, seed=3)
    assert a == b
    # Identical runs: a paired contrast has zero spread; unpaired resampling would not.
    same = {"base": runs["base"], "arm": runs["base"]}
    g = analyze.analyze(same, "base", {"aff": "arm"}, resamples=1000)["gate"]["aff"]["aligned_rate_all"]
    assert g["ci"] == [0.0, 0.0]


def test_ci_width_matches_binomial_theory(tmp_path) -> None:
    # Responses here are independent within a question, so the clustered interval
    # should match the iid binomial width: 2 * 1.96 * sqrt(0.4 * 0.6 / 600) = 0.078.
    runs = _gate_runs(tmp_path, 300, 0.4, 0.4)
    res = analyze.analyze(runs, "base", {}, resamples=4000)
    m = res["by_eval"]["aff"]["adapters"]["arm"]["aligned_rate_all"]
    assert m["ci"][0] < m["value"] < m["ci"][1]
    assert 0.065 < m["ci"][1] - m["ci"][0] < 0.095


def test_mismatched_questions_and_bad_labels_raise(tmp_path) -> None:
    base = _run(tmp_path, "base", "aff", {"q0": ("aligned", "aligned"), "q1": ("aligned", "aligned")})
    other = _run(tmp_path, "other", "aff", {"q0": ("aligned", "aligned"), "q9": ("aligned", "aligned")})
    with pytest.raises(SystemExit, match="differ in their aff questions"):
        analyze.analyze({"base": base, "x": other}, "base", {})
    with pytest.raises(SystemExit, match="is not one of the labels"):
        analyze.analyze({"base": base}, "nope", {})
    with pytest.raises(SystemExit, match="not one of the labels"):
        analyze.analyze({"base": base}, "base", {"aff": "nope"})
    with pytest.raises(SystemExit, match="KEY=VALUE"):
        analyze._pairs(["novalue"], "--gate")


def test_unjudged_records_rejected(tmp_path) -> None:
    path = _write(tmp_path, "u", [("aff", "q0", "orig", "unjudged")])
    with pytest.raises(SystemExit, match="unjudged"):
        analyze.load_counts(path, "pooled")


def test_cli_writes_results(tmp_path) -> None:
    runs = _gate_runs(tmp_path, 50, 0.7, 0.1)
    out = tmp_path / "res" / "results.json"
    # Repeated flags accumulate, as the launcher passes lists.
    assert analyze.main(["--labels", "base", "--labels", "arm", "--responses", runs["base"], "--responses", runs["arm"],
                         "--baseline", "base", "--gate", "aff=arm", "--resamples", "500", "--out", str(out)]) == 0
    res = json.loads(out.read_text())
    assert res["gate_passes"] and res["config"]["ci"].startswith("percentile")


def test_cli_rejects_mismatched_labels(tmp_path) -> None:
    runs = _gate_runs(tmp_path, 10, 0.5, 0.5)
    with pytest.raises(SystemExit, match="--labels but"):
        analyze.main(["--labels", "base", "arm", "x", "--responses", runs["base"], runs["arm"],
                      "--baseline", "base", "--out", str(tmp_path / "r.json")])
