"""Seed-level analysis: arm means over training seeds, contrasts with seed-level CIs.

Implements ``docs/preregistration/phase2b-section31-seeds.md`` §6. ``analyze`` and
``analyze_logprob`` treat each run as fixed and give question-sampling intervals only;
here each arm is several training runs (seeds) on the same questions, and the
uncertainty combines seed-to-seed variation with question sampling.

* **Inputs.** One run per (arm, seed): ``--labels`` (arm), ``--seeds`` and ``--runs``,
  matched by position (separate lists because the launcher resolves a ``file:``
  reference only at the start of an argument). ``--kind judge`` reads judged
  ``preference.jsonl`` files (``aligned_rate_all``, ratio of sums as in ``analyze``);
  ``--kind logprob`` reads ``logprob.jsonl`` files (mean ``p_aligned``, as in
  ``analyze_logprob``). An arm with a single run (e.g. a released adapter) is *fixed*:
  it is paired with every seed of the other arm and adds no seed variance.
* **Contrast** ``--contrast NAME=EVAL:A,B``: per seed s, ``c_s = rate(A_s) - rate(B_s)``
  over the seeds both arms have (pairing by seed number; for arms sharing an MSM stage
  this pairs runs that share it). Point estimate: the mean of ``c_s``.
  - ``sd_seed``: standard deviation of ``c_s`` over seeds (n - 1 denominator).
  - ``se_question``: bootstrap standard deviation of the mean contrast when questions
    are resampled (question-clustered, one resample matrix per eval set shared by all
    runs, the seeding scheme of ``analyze``). The questions are the same for every seed,
    so ``sd_seed`` does not contain question-sampling variance.
  - **CI:** mean ± t(0.975, n - 1) · sqrt(sd_seed² / n + se_question²), n = number of
    seeds. With one seed (both arms fixed), the question-bootstrap percentile CI.
  - **Prediction interval** for one new seed, on these questions:
    mean ± t(0.975, n - 1) · sd_seed · sqrt(1 + 1/n). No question term, because the
    value it is compared with (a released adapter) is measured on the same questions.
* **Arm summary:** the same quantities for each arm's rate (per-seed values, mean,
  ``sd_seed``, CI, range, prediction interval).
* ``--within NAME=EVAL:QUANT~REF``: whether a reference quantity lies within our seed
  spread. ``QUANT`` and ``REF`` are each an arm ``X`` or a contrast ``X,Y`` on EVAL;
  reports whether REF's value is inside QUANT's prediction interval and inside the range
  of its per-seed values.
* **Gate:** ``--gate-contrast NAME`` (seed-level CI lower bound > 0) and
  ``--gate-within NAME`` (reference inside the prediction interval); the gate passes if
  every listed item passes.

Usage::

    python -m msm_repro.analyze_seeds --kind judge --labels a a b b --seeds 0 1 0 1 \\
        --runs r/a0/preference.jsonl r/a1/preference.jsonl r/b0/preference.jsonl r/b1/preference.jsonl \\
        --contrast D=affordability:a,b --gate-contrast D --out results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from .analyze import _ci, _pairs, load_counts, parse_contrast
    from .analyze_logprob import load_scores
except ImportError:  # executed as a script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from msm_repro.analyze import _ci, _pairs, load_counts, parse_contrast
    from msm_repro.analyze_logprob import load_scores


# Two-sided 95% Student-t quantiles, t(0.975, df). A table instead of a scipy dependency;
# 4 seeds give df = 3.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def _load(kind: str, path: str) -> Dict[str, Dict[str, Any]]:
    """{eval: {"ids": [...], "point": float, "boot": callable(idx) -> array}}."""
    out = {}
    if kind == "judge":
        for ev, c in load_counts(path, "pooled").items():
            out[ev] = {"ids": c["ids"], "num": c["aligned"], "den": c["n"]}
    elif kind == "logprob":
        for ev, s in load_scores(path).items():
            out[ev] = {"ids": s["ids"], "num": s["p_aligned"], "den": np.ones_like(s["p_aligned"])}
    else:
        raise SystemExit(f"--kind must be judge or logprob, got {kind!r}")
    return out


def _value(d: Dict[str, Any], idx: Optional[np.ndarray] = None):
    if idx is None:
        return float(d["num"].sum() / d["den"].sum())
    return d["num"][idx].sum(axis=1) / d["den"][idx].sum(axis=1)


def _summary(per_seed: Dict[int, float], boot_mean: np.ndarray, level: float) -> Dict[str, Any]:
    vals = np.array(list(per_seed.values()))
    n = len(vals)
    mean = float(vals.mean())
    se_q = float(boot_mean.std(ddof=1))
    out: Dict[str, Any] = {"per_seed": {str(s): v for s, v in per_seed.items()}, "n_seeds": n, "mean": mean,
                           "se_question": se_q, "range": [float(vals.min()), float(vals.max())]}
    if n >= 2:
        sd = float(vals.std(ddof=1))
        if level != 0.95 or n - 1 not in T975:
            raise SystemExit(f"seed-level intervals support level 0.95 and 2-11 seeds, got level {level}, {n} seeds")
        t = T975[n - 1]
        half = t * float(np.sqrt(sd ** 2 / n + se_q ** 2))
        pi = t * sd * float(np.sqrt(1 + 1 / n))
        out.update({"sd_seed": sd, "t": t, "se_combined": float(np.sqrt(sd ** 2 / n + se_q ** 2)),
                    "ci": [mean - half, mean + half], "ci_kind": "seed-level t (seed + question)",
                    "prediction_interval": [mean - pi, mean + pi]})
    else:
        out.update({"sd_seed": None, "ci": _ci(boot_mean, level), "ci_kind": "question bootstrap (one seed)",
                    "prediction_interval": None})
    out["lower_bound_above_zero"] = out["ci"][0] > 0
    out["upper_bound_below_zero"] = out["ci"][1] < 0
    return out


def _parse_quantity(spec: str) -> Tuple[str, ...]:
    parts = tuple(spec.split(","))
    if len(parts) not in (1, 2) or not all(parts):
        raise SystemExit(f"expected an arm X or a contrast X,Y, got {spec!r}")
    return parts


def analyze_seeds(runs: Sequence[Tuple[str, int, str]], kind: str = "judge",
                  contrasts: Optional[Dict[str, str]] = None, within: Optional[Dict[str, str]] = None,
                  gate_contrasts: Sequence[str] = (), gate_within: Sequence[str] = (),
                  resamples: int = 10_000, seed: int = 0, level: float = 0.95) -> Dict[str, Any]:
    arms: Dict[str, Dict[int, str]] = {}
    for label, s, path in runs:
        if s in arms.setdefault(label, {}):
            raise SystemExit(f"arm {label!r} has seed {s} twice")
        arms[label][s] = path
    data = {(a, s): _load(kind, p) for a, ss in arms.items() for s, p in ss.items()}
    ref_key = next(iter(data))
    evals = sorted(data[ref_key])
    for k, d in data.items():
        if sorted(d) != evals or any(d[ev]["ids"] != data[ref_key][ev]["ids"] for ev in evals):
            raise SystemExit(f"run {k} and {ref_key} differ in their questions")
    contrasts = {n: parse_contrast(c) for n, c in (contrasts or {}).items()}
    for n, (ev, a, b) in contrasts.items():
        if ev not in evals or a not in arms or b not in arms:
            raise SystemExit(f"--contrast {n}: unknown eval set or arm")
    within_p = {}
    for n, spec in (within or {}).items():
        ev, sep, rest = spec.partition(":")
        q, sep2, r = rest.partition("~")
        if not (sep and sep2) or ev not in evals:
            raise SystemExit(f"--within {n}: expected EVAL:QUANT~REF, got {spec!r}")
        q, r = _parse_quantity(q), _parse_quantity(r)
        if len(q) != len(r) or any(x not in arms for x in q + r):
            raise SystemExit(f"--within {n}: unknown arm, or arm compared with a contrast")
        within_p[n] = (ev, q, r)
    for n in gate_contrasts:
        if n not in contrasts:
            raise SystemExit(f"--gate-contrast {n!r} is not a named --contrast")
    for n in gate_within:
        if n not in within_p:
            raise SystemExit(f"--gate-within {n!r} is not a named --within")

    rng = np.random.default_rng(seed)
    point, boot = {}, {}
    res: Dict[str, Any] = {
        "config": {"kind": kind, "runs": {a: {str(s): p for s, p in ss.items()} for a, ss in arms.items()},
                   "contrasts": {n: f"{e}:{a},{b}" for n, (e, a, b) in contrasts.items()},
                   "within": dict(within or {}), "gate_contrasts": list(gate_contrasts),
                   "gate_within": list(gate_within), "resamples": resamples, "seed": seed, "level": level,
                   "metric": "aligned_rate_all" if kind == "judge" else "mean_p_aligned"},
        "by_eval": {}, "contrasts": {}, "within": {}}
    for ev in evals:  # sorted, so the RNG stream per eval set is fixed
        n_q = len(data[ref_key][ev]["ids"])
        idx = rng.integers(0, n_q, size=(resamples, n_q))
        for k, d in data.items():
            point[k, ev] = _value(d[ev])
            boot[k, ev] = _value(d[ev], idx)
        res["by_eval"][ev] = {"n_questions": n_q, "arms": {
            a: _summary({s: point[(a, s), ev] for s in sorted(ss)},
                        np.mean([boot[(a, s), ev] for s in ss], axis=0), level)
            for a, ss in arms.items()}}

    def quantity(ev: str, q: Tuple[str, ...]):
        """Per-seed values and the bootstrap of their mean, for an arm or a contrast."""
        if len(q) == 1:
            ss = sorted(arms[q[0]])
            return ({s: point[(q[0], s), ev] for s in ss}, np.mean([boot[(q[0], s), ev] for s in ss], axis=0))
        a, b = q
        sa, sb = sorted(arms[a]), sorted(arms[b])
        if len(sa) == 1 or len(sb) == 1:
            seeds = sb if len(sa) == 1 else sa
            pair = [(sa[0] if len(sa) == 1 else s, sb[0] if len(sb) == 1 else s, s) for s in seeds]
        else:
            common = sorted(set(sa) & set(sb))
            if not common:
                raise SystemExit(f"arms {a!r} and {b!r} share no seed")
            pair = [(s, s, s) for s in common]
        per = {s: point[(a, x), ev] - point[(b, y), ev] for x, y, s in pair}
        bm = np.mean([boot[(a, x), ev] - boot[(b, y), ev] for x, y, _ in pair], axis=0)
        return per, bm

    for n, (ev, a, b) in contrasts.items():
        per, bm = quantity(ev, (a, b))
        res["contrasts"][n] = {"eval": ev, "a": a, "b": b, "gate": n in gate_contrasts, **_summary(per, bm, level)}
    for n, (ev, q, r) in within_p.items():
        ours = _summary(*quantity(ev, q), level)
        ref_per, _ = quantity(ev, r)
        ref = float(np.mean(list(ref_per.values())))
        pi = ours["prediction_interval"]
        res["within"][n] = {"eval": ev, "ours": ",".join(q), "reference": ",".join(r), "gate": n in gate_within,
                            "ours_mean": ours["mean"], "ours_per_seed": ours["per_seed"], "ours_sd_seed": ours["sd_seed"],
                            "prediction_interval": pi, "range": ours["range"], "reference_value": ref,
                            "reference_minus_mean": ref - ours["mean"],
                            "in_prediction_interval": bool(pi is not None and pi[0] <= ref <= pi[1]),
                            "in_range": bool(ours["range"][0] <= ref <= ours["range"][1])}
    if gate_contrasts or gate_within:
        verdicts = [res["contrasts"][n]["lower_bound_above_zero"] for n in gate_contrasts]
        verdicts += [res["within"][n]["in_prediction_interval"] for n in gate_within]
        res["gate_passes"] = all(verdicts)
    return res


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", choices=["judge", "logprob"], default="judge")
    p.add_argument("--labels", nargs="+", action="extend", required=True, help="arm label per run")
    p.add_argument("--seeds", nargs="+", action="extend", required=True, type=int, help="training seed per run")
    p.add_argument("--runs", nargs="+", action="extend", required=True,
                   help="preference.jsonl (judge) or logprob.jsonl (logprob) per run, in the same order")
    p.add_argument("--contrast", action="append", default=[], metavar="NAME=EVAL:A,B")
    p.add_argument("--within", action="append", default=[], metavar="NAME=EVAL:QUANT~REF")
    p.add_argument("--gate-contrast", action="append", default=[], metavar="NAME")
    p.add_argument("--gate-within", action="append", default=[], metavar="NAME")
    p.add_argument("--resamples", type=int, default=10_000)
    p.add_argument("--level", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not len(args.labels) == len(args.seeds) == len(args.runs):
        raise SystemExit(f"{len(args.labels)} --labels, {len(args.seeds)} --seeds, {len(args.runs)} --runs: must match")
    res = analyze_seeds(list(zip(args.labels, args.seeds, args.runs)), args.kind,
                        _pairs(args.contrast, "--contrast"), _pairs(args.within, "--within"),
                        args.gate_contrast, args.gate_within, args.resamples, args.seed, args.level)
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)

    def fmt(x: Dict[str, Any]) -> str:
        sd = f" sd_seed {x['sd_seed']:.3f}" if x["sd_seed"] is not None else ""
        return f"{x['mean']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}] (n={x['n_seeds']}{sd})"

    for ev, e in res["by_eval"].items():
        print(f"{ev} ({e['n_questions']} questions)")
        for a, x in e["arms"].items():
            print(f"  {a:<40} {fmt(x)}")
    for n, c in res["contrasts"].items():
        flag = "above 0" if c["lower_bound_above_zero"] else "below 0" if c["upper_bound_below_zero"] else "includes 0"
        print(f"contrast {n} ({c['eval']}): {c['a']} - {c['b']} = {fmt(c)} CI {flag}{' [gate]' if c['gate'] else ''}")
    for n, w in res["within"].items():
        pi = w["prediction_interval"]
        pis = f"[{pi[0]:+.3f}, {pi[1]:+.3f}]" if pi else "n/a"
        print(f"within {n} ({w['eval']}): reference {w['reference_value']:+.3f} vs ours {w['ours_mean']:+.3f}, "
              f"PI {pis} {'in' if w['in_prediction_interval'] else 'OUT'}, range {'in' if w['in_range'] else 'out'}"
              f"{' [gate]' if w['gate'] else ''}")
    if "gate_passes" in res:
        print("GATE", "PASSES" if res["gate_passes"] else "FAILS")
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
