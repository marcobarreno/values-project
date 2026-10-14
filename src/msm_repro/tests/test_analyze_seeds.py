"""Unit tests for the seed-level analysis (Phase 2b): per-seed contrasts, combined CIs, within-spread checks."""

from __future__ import annotations

import json

import numpy as np
import pytest

from msm_repro import analyze_seeds as asd


def _judge(tmp_path, name, p, n_q=100, ev="aff", seed=0):
    """A judged run whose questions are aligned with probability p (both orders the same)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_q):
        s = "aligned" if rng.random() < p else "misaligned"
        rows += [{"eval": ev, "id": f"q{i:03d}", "variant": v, "status": s} for v in ("orig", "swapped")]
    path = tmp_path / f"{name}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def _fixed_rate(tmp_path, name, k, n_q=100, ev="aff"):
    """First k questions aligned in both orders, the rest misaligned: rate exactly k / n_q."""
    rows = []
    for i in range(n_q):
        s = "aligned" if i < k else "misaligned"
        rows += [{"eval": ev, "id": f"q{i:03d}", "variant": v, "status": s} for v in ("orig", "swapped")]
    path = tmp_path / f"{name}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def test_per_seed_contrast_mean_sd_and_t_interval(tmp_path) -> None:
    # A: rates 0.50, 0.60, 0.70, 0.80 over seeds; B: 0.30 for every seed. Contrasts 0.2..0.5.
    runs = [("a", s, _fixed_rate(tmp_path, f"a{s}", 50 + 10 * s)) for s in range(4)]
    runs += [("b", s, _fixed_rate(tmp_path, f"b{s}", 30)) for s in range(4)]
    res = asd.analyze_seeds(runs, contrasts={"D": "aff:a,b"}, resamples=500)
    c = res["contrasts"]["D"]
    assert c["per_seed"] == {"0": pytest.approx(0.2), "1": pytest.approx(0.3), "2": pytest.approx(0.4), "3": pytest.approx(0.5)}
    assert c["mean"] == pytest.approx(0.35) and c["n_seeds"] == 4
    sd = float(np.std([0.2, 0.3, 0.4, 0.5], ddof=1))
    assert c["sd_seed"] == pytest.approx(sd)
    se = np.sqrt(sd ** 2 / 4 + c["se_question"] ** 2)
    assert c["ci"] == pytest.approx([0.35 - 3.182 * se, 0.35 + 3.182 * se])
    assert c["prediction_interval"] == pytest.approx([0.35 - 3.182 * sd * np.sqrt(1.25), 0.35 + 3.182 * sd * np.sqrt(1.25)])
    assert c["range"] == pytest.approx([0.2, 0.5]) and c["lower_bound_above_zero"]
    arm = res["by_eval"]["aff"]["arms"]["a"]
    assert arm["mean"] == pytest.approx(0.65) and arm["per_seed"]["3"] == pytest.approx(0.8)


def test_seed_variance_widens_the_interval_beyond_question_sampling(tmp_path) -> None:
    # Same question-level noise, but one pair of arms has large seed-to-seed swings.
    calm = [("a", s, _judge(tmp_path, f"ca{s}", 0.5, seed=s)) for s in range(4)]
    calm += [("b", s, _judge(tmp_path, f"cb{s}", 0.3, seed=10 + s)) for s in range(4)]
    wild = [("a", s, _judge(tmp_path, f"wa{s}", p, seed=s)) for s, p in enumerate([0.2, 0.8, 0.3, 0.7])]
    wild += [("b", s, _judge(tmp_path, f"wb{s}", 0.3, seed=10 + s)) for s in range(4)]
    w_calm = np.diff(asd.analyze_seeds(calm, contrasts={"D": "aff:a,b"}, resamples=500)["contrasts"]["D"]["ci"])[0]
    w_wild = np.diff(asd.analyze_seeds(wild, contrasts={"D": "aff:a,b"}, resamples=500)["contrasts"]["D"]["ci"])[0]
    assert w_wild > 2 * w_calm


def test_fixed_arm_is_paired_with_every_seed_and_within_checks(tmp_path) -> None:
    runs = [("ours_a", s, _fixed_rate(tmp_path, f"a{s}", k)) for s, k in enumerate([40, 50, 45, 55])]
    runs += [("ours_b", s, _fixed_rate(tmp_path, f"b{s}", 20)) for s in range(4)]
    runs += [("rel_a", 0, _fixed_rate(tmp_path, "ra", 48)), ("rel_b", 0, _fixed_rate(tmp_path, "rb", 20)),
             ("far", 0, _fixed_rate(tmp_path, "far", 95))]
    res = asd.analyze_seeds(runs, contrasts={"D": "aff:ours_a,rel_a"},
                            within={"gap": "aff:ours_a,ours_b~rel_a,rel_b", "arm": "aff:ours_a~rel_a",
                                    "out": "aff:ours_a~far"},
                            gate_within=["gap"], resamples=500)
    d = res["contrasts"]["D"]
    assert d["n_seeds"] == 4 and d["per_seed"]["0"] == pytest.approx(-0.08) and d["mean"] == pytest.approx(-0.005)
    g = res["within"]["gap"]
    assert g["reference_value"] == pytest.approx(0.28) and g["ours_mean"] == pytest.approx(0.275)
    assert g["in_prediction_interval"] and g["in_range"] and res["gate_passes"]
    assert res["within"]["arm"]["in_range"]
    o = res["within"]["out"]
    assert not o["in_prediction_interval"] and not o["in_range"]


def test_single_seed_contrast_falls_back_to_question_bootstrap(tmp_path) -> None:
    runs = [("a", 0, _judge(tmp_path, "a", 0.7)), ("b", 0, _judge(tmp_path, "b", 0.2, seed=1))]
    c = asd.analyze_seeds(runs, contrasts={"D": "aff:a,b"}, resamples=500)["contrasts"]["D"]
    assert c["sd_seed"] is None and c["prediction_interval"] is None
    assert c["ci_kind"].startswith("question") and c["ci"][0] < c["mean"] < c["ci"][1]


def test_logprob_kind(tmp_path) -> None:
    def scores(name, ps):
        rows = [{"eval": "aff", "id": f"q{i}", "variant": v, "p_aligned": p, "letter_mass": 0.9, "p_A_norm": 0.5}
                for i, p in enumerate(ps) for v in ("orig", "swapped")]
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return str(path)
    runs = [("a", s, scores(f"a{s}", [0.6 + 0.01 * s] * 5)) for s in range(3)]
    runs += [("b", s, scores(f"b{s}", [0.4] * 5)) for s in range(3)]
    c = asd.analyze_seeds(runs, kind="logprob", contrasts={"D": "aff:a,b"}, resamples=200)["contrasts"]["D"]
    assert c["mean"] == pytest.approx(0.21) and c["se_question"] == pytest.approx(0.0, abs=1e-12)


def test_bad_inputs_raise(tmp_path) -> None:
    a0, b0 = _judge(tmp_path, "a0", 0.5), _judge(tmp_path, "b0", 0.5, n_q=90)
    with pytest.raises(SystemExit):  # different questions
        asd.analyze_seeds([("a", 0, a0), ("b", 0, b0)])
    with pytest.raises(SystemExit):  # duplicate seed
        asd.analyze_seeds([("a", 0, a0), ("a", 0, a0)])
    a1 = _judge(tmp_path, "a1", 0.5, seed=1)
    with pytest.raises(SystemExit):  # unknown arm
        asd.analyze_seeds([("a", 0, a0), ("a", 1, a1)], contrasts={"D": "aff:a,zzz"})
    with pytest.raises(SystemExit):  # arm compared with a contrast
        asd.analyze_seeds([("a", 0, a0), ("a", 1, a1)], within={"W": "aff:a~a,a"})
    with pytest.raises(SystemExit):  # no shared seed
        asd.analyze_seeds([("a", 0, a0), ("a", 1, a1), ("b", 2, _judge(tmp_path, "b2", 0.5)),
                           ("b", 3, _judge(tmp_path, "b3", 0.5))], contrasts={"D": "aff:a,b"})


def test_cli(tmp_path, capsys) -> None:
    paths = [_fixed_rate(tmp_path, f"a{s}", 50 + 5 * s) for s in range(2)] + [_fixed_rate(tmp_path, f"b{s}", 20) for s in range(2)]
    out = tmp_path / "res" / "results.json"
    asd.main(["--labels", "a", "a", "b", "b", "--seeds", "0", "1", "0", "1", "--runs", *paths,
              "--contrast", "D=aff:a,b", "--gate-contrast", "D", "--resamples", "200", "--out", str(out)])
    res = json.loads(out.read_text())
    assert res["contrasts"]["D"]["mean"] == pytest.approx(0.325) and "gate_passes" in res
    assert "GATE" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        asd.main(["--labels", "a", "--seeds", "0", "1", "--runs", paths[0], "--out", str(out)])
