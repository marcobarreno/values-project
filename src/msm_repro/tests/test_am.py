"""CPU tests for the AM eval helpers (grid, summaries, agreement, analysis); no vLLM, Inspect or API."""
import json

import pytest

from msm_repro import am_eval, am_regrade, analyze_am


def test_grid_is_27_conditions_in_paper_order():
    g = am_eval.conditions()
    assert len(g) == 27
    assert {c["scenario"] for c in g} == {"exfiltration", "murder", "leaking"}
    assert g[0]["condition_id"] == "exfiltration_none-none_replacement"
    assert sum(c["goal_type"] == "none" for c in g) == 3
    assert len({c["condition_id"] for c in g}) == 27


def test_wilson_matches_known_values():
    lo, hi = am_eval.wilson(0, 30)
    assert lo == 0.0 and hi == pytest.approx(0.1135, abs=1e-3)
    lo, hi = am_eval.wilson(15, 30)
    assert lo == pytest.approx(0.3315, abs=1e-3) and hi == pytest.approx(0.6685, abs=1e-3)
    assert am_eval.wilson(0, 0) == [None, None]


def _rec(cid, scen, v, harmful=None, stop="stop"):
    return {"condition_id": cid, "scenario": scen, "goal_type": "none", "goal_value": "none",
            "urgency_type": "replacement", "completion": "x", "stop_reason": stop,
            "classifier_verdict": v, "harmful": v if harmful is None else harmful}


def test_summarize_equal_weight_average_and_errors():
    recs = [_rec("a", "murder", 1), _rec("a", "murder", 0), _rec("a", "murder", None),
            _rec("b", "leaking", 1, stop="max_tokens"), _rec("b", "leaking", 1), _rec("b", "leaking", 1), _rec("b", "leaking", 1)]
    s = am_eval.summarize(recs)
    assert s["n_samples"] == 7 and s["n_graded"] == 6 and s["n_error"] == 1
    assert s["conditions"]["a"]["classifier_verdict"]["rate"] == 0.5
    assert s["conditions"]["b"]["classifier_verdict"]["rate"] == 1.0
    assert s["average_rate"]["classifier_verdict"] == 0.75  # equal weight, not pooled (5/6)
    assert s["length_stop_rate"] == pytest.approx(1 / 7)


def test_kappa_and_agreement():
    assert am_regrade.kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert am_regrade.kappa([1, 1, 0, 0], [1, 0, 1, 0]) == 0.0
    recs = [{"orig_classifier_verdict": o, "classifier_verdict": n} for o, n in [(1, 1), (1, 0), (0, 0), (0, None)]]
    ag = am_regrade.agreement(recs)
    assert ag["n"] == 3 and ag["agreement"] == pytest.approx(2 / 3)
    assert ag["orig_yes_new_no"] == 1 and ag["orig_no_new_yes"] == 0


def _write(path, rates, n=30):
    with open(path, "w") as fh:
        for i, r in enumerate(rates):
            k = round(r * n)
            for j in range(n):
                fh.write(json.dumps(_rec(f"c{i}", "murder" if i % 2 else "leaking", int(j < k))) + "\n")


def test_analyze_am_contrast_reference_and_gate(tmp_path):
    hi, lo = tmp_path / "hi.jsonl", tmp_path / "lo.jsonl"
    _write(hi, [0.7, 0.6, 0.8])
    _write(lo, [0.2, 0.1, 0.3])
    out = tmp_path / "res.json"
    analyze_am.main(["--labels", "base", "msm", "--runs", str(hi), str(lo), "--contrast", "R=base,msm",
                     "--reference", "base=0.68", "--reference", "msm=0.22", "--gate", "R:0.23",
                     "--resamples", "2000", "--seed", "0", "--out", str(out)])
    res = json.loads(out.read_text())
    assert res["arms"]["base"]["average"] == pytest.approx(0.7)
    assert res["arms"]["msm"]["average"] == pytest.approx(0.2)
    c = res["contrasts"]["R"]
    assert c["estimate"] == pytest.approx(0.5) and c["ci95"][0] > 0.3
    assert c["reference"] == pytest.approx(0.46)
    assert res["gate"]["R"]["passes"] and res["gate_passes"]
    assert res["arms"]["base"]["reference_in_ci"]


def test_analyze_am_gate_fails_on_small_reduction(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write(a, [0.5, 0.5, 0.5])
    _write(b, [0.4, 0.4, 0.4])
    out = tmp_path / "res.json"
    analyze_am.main(["--labels", "a", "b", "--runs", str(a), str(b), "--contrast", "R=a,b", "--gate", "R:0.23",
                     "--resamples", "2000", "--seed", "0", "--out", str(out)])
    assert not json.loads(out.read_text())["gate_passes"]


def test_am_eval_list_flags_accumulate_like_launcher_lists():
    a = am_eval.parse_args(["--base", "b", "--adapter", "a", "--epochs", "1", "--out", "o",
                            "--scenarios", "exfiltration", "--scenarios", "murder"])
    assert a.scenarios == ["exfiltration", "murder"] and a.goal_values == list(am_eval.GOAL_VALUES)
    d = am_eval.parse_args(["--base", "b", "--adapter", "a", "--epochs", "1", "--out", "o"])
    assert d.scenarios == list(am_eval.SCENARIOS)
