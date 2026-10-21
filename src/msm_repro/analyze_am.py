"""Analyze AM eval runs: average misalignment rates, contrasts and the Phase 4a gate (no GPU, no API).

Inputs: one ``am_eval`` ``am.jsonl`` per arm (``--labels`` and ``--runs`` matched by position),
each a single fixed run (one released adapter, one training seed).

* **Per arm:** the equal-weight average over conditions of the per-condition rate of the metric
  (default ``classifier_verdict``, the paper's), per scenario, per condition (Wilson 95%).
* **Uncertainty:** transcript sampling only. 95% percentile intervals from ``--resamples``
  stratified bootstrap resamples (transcripts resampled with replacement within each condition,
  seed ``--seed``); arms are independent samples, so a contrast resamples each arm separately.
  This interval does not include training-seed variation: the paper's values are 4-seed means.
* **Contrast** ``--contrast NAME=A,B``: average(A) - average(B).
* **Reference** ``--reference LABEL=VALUE``: a published value per arm, reported beside ours with
  whether it lies inside our interval (descriptive).
* **Gate** ``--gate NAME:MIN``: contrast NAME passes if its interval lower bound is > 0 and its
  point estimate is >= MIN. The run passes if every listed gate passes.

Records with no verdict (sample or grader errors) are dropped from their condition and counted.
Output: ``results.json`` (``--out``) and a printed report.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def load(path: str, metric: str) -> Tuple[Dict[str, np.ndarray], Dict[str, str], int]:
    by: Dict[str, List[int]] = defaultdict(list)
    scen: Dict[str, str] = {}
    dropped = 0
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        scen[r["condition_id"]] = r["scenario"]
        if r.get(metric) is None:
            dropped += 1
            by.setdefault(r["condition_id"], [])
            continue
        by[r["condition_id"]].append(int(r[metric]))
    return {k: np.array(v, dtype=float) for k, v in sorted(by.items())}, scen, dropped


def boot_averages(cond: Dict[str, np.ndarray], resamples: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap distribution of the equal-weight average over conditions (stratified by condition)."""
    keys = [k for k, v in cond.items() if len(v)]
    tot = np.zeros(resamples)
    for k in keys:
        v = cond[k]
        idx = rng.integers(0, len(v), size=(resamples, len(v)))
        tot += v[idx].mean(axis=1)
    return tot / len(keys)


def wilson(k: float, n: int, z: float = 1.959964) -> List[Optional[float]]:
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", nargs="+", action="extend", required=True)
    p.add_argument("--runs", nargs="+", action="extend", required=True, help="am.jsonl per label")
    p.add_argument("--metric", default="classifier_verdict", choices=["classifier_verdict", "harmful"])
    p.add_argument("--contrast", action="append", default=[], help="NAME=A,B (A minus B)")
    p.add_argument("--reference", action="append", default=[], help="LABEL=VALUE (published average rate)")
    p.add_argument("--gate", action="append", default=[], help="NAME:MIN for a contrast")
    p.add_argument("--resamples", type=int, default=10000)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = parse_args(argv)
    if len(a.labels) != len(a.runs):
        raise SystemExit("--labels and --runs must have the same length")
    rng = np.random.default_rng(a.seed)
    refs = {k: float(v) for k, v in (r.split("=", 1) for r in a.reference)}
    arms, boots = {}, {}
    for lab, path in zip(a.labels, a.runs):
        cond, scen, dropped = load(path, a.metric)
        rates = {k: float(v.mean()) for k, v in cond.items() if len(v)}
        avg = float(np.mean(list(rates.values())))
        boots[lab] = boot_averages(cond, a.resamples, rng)
        ci = [float(x) for x in np.percentile(boots[lab], [2.5, 97.5])]
        by_s = defaultdict(list)
        for k, r in rates.items():
            by_s[scen[k]].append(r)
        arms[lab] = {
            "run": path, "n_conditions": len(rates), "n_graded": int(sum(len(v) for v in cond.values())),
            "n_dropped": dropped, "average": avg, "ci95": ci,
            "by_scenario": {s: float(np.mean(v)) for s, v in sorted(by_s.items())},
            "conditions": {k: {"n": int(len(v)), "rate": float(v.mean()) if len(v) else None,
                               "wilson95": wilson(float(v.sum()), len(v))} for k, v in cond.items()},
        }
        if lab in refs:
            arms[lab]["reference"] = refs[lab]
            arms[lab]["reference_in_ci"] = bool(ci[0] <= refs[lab] <= ci[1])
    contrasts = {}
    for spec in a.contrast:
        name, pair = spec.split("=", 1)
        x, y = pair.split(",")
        d = boots[x] - boots[y]
        est = arms[x]["average"] - arms[y]["average"]
        ci = [float(v) for v in np.percentile(d, [2.5, 97.5])]
        contrasts[name] = {"a": x, "b": y, "estimate": est, "ci95": ci}
        if x in refs and y in refs:
            contrasts[name]["reference"] = refs[x] - refs[y]
            contrasts[name]["reference_in_ci"] = bool(ci[0] <= refs[x] - refs[y] <= ci[1])
    gates = {}
    for g in a.gate:
        name, mn = g.split(":")
        c = contrasts[name]
        gates[name] = {"min_estimate": float(mn), "lower_bound_above_zero": c["ci95"][0] > 0,
                       "estimate_at_least_min": c["estimate"] >= float(mn)}
        gates[name]["passes"] = gates[name]["lower_bound_above_zero"] and gates[name]["estimate_at_least_min"]
    res = {"config": vars(a), "metric": a.metric, "arms": arms, "contrasts": contrasts, "gate": gates,
           "gate_passes": all(g["passes"] for g in gates.values()) if gates else None}
    json.dump(res, open(a.out, "w"), indent=1)
    for lab, r in arms.items():
        ref = f"  paper {r['reference']:.3f} ({'inside' if r['reference_in_ci'] else 'OUTSIDE'})" if "reference" in r else ""
        print(f"{lab:24} average {r['average']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]  n={r['n_graded']} "
              f"dropped={r['n_dropped']}  " + " ".join(f"{s} {v:.3f}" for s, v in r["by_scenario"].items()) + ref)
    for n, c in contrasts.items():
        ref = f"  paper {c['reference']:+.3f}" if "reference" in c else ""
        print(f"contrast {n}: {c['a']} - {c['b']} = {c['estimate']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]{ref}")
    for n, g in gates.items():
        print(f"gate {n}: {'PASS' if g['passes'] else 'FAIL'} (CI>0 {g['lower_bound_above_zero']}, >= {g['min_estimate']} {g['estimate_at_least_min']})")
    if gates:
        print("GATE PASSES" if res["gate_passes"] else "GATE FAILS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
