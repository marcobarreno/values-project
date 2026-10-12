"""Aligned rates with question-clustered bootstrap CIs, and the Phase 1 gate contrasts.

Implements the analysis pre-registered in ``docs/preregistration/phase1-section31.md``
§6. Inputs are judged ``preference.jsonl`` files from ``eval_preference.py`` or
``rescore.py``, one per adapter, all on the same questions.

* **Clustering.** Every response to a question (both question orders, all samples)
  is one cluster. The bootstrap resamples questions with replacement; a rate is
  the ratio of sums over the resampled questions.
* **Pairing.** Within an eval set the same resampled questions are used for every
  adapter, so a contrast between two adapters is a paired bootstrap.
* **Gate.** For each ``--gate EVAL=ARM`` the contrast is
  ``rate(ARM) - rate(--baseline)`` on that eval set. It passes if the lower bound
  of its CI is above 0; the gate passes if every contrast passes.
* **Named contrasts** (Phase 2a, ``docs/preregistration/phase2a-section31-training.md``
  §6). ``--contrast NAME=EVAL:A,B`` is ``rate(A) - rate(B)`` on EVAL, for any two
  labels. ``--difference NAME=C1,C2`` is contrast C1 minus contrast C2 (both on the
  same eval set), e.g. our gap minus the released adapters' gap. Each reports its CI
  and whether the CI lies entirely above or entirely below 0. ``--gate-contrast NAME``
  adds a named contrast to the gate verdict.

Reported per adapter and eval set: ``aligned_rate_all`` (primary),
``aligned_rate_decided``, ``decided_rate``, each with a percentile CI, and the
position gap (``aligned_rate_all`` on original minus swapped question order,
paired by question) when both orders are present.

Usage::

    python -m msm_repro.analyze --labels baseline x --responses runs/a/preference.jsonl runs/b/preference.jsonl \\
        --baseline baseline --gate affordability=x --orders pooled --out results.json

    python -m msm_repro.analyze --labels a1 a2 --responses ... --baseline a1 \\
        --contrast H1=affordability:a2,a1 --gate-contrast H1 --out results.json

Labels and paths are matched by position. Both flags accumulate when repeated, which is
how the launcher passes lists (a ``file:`` reference must start an argument to be resolved).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

METRICS = ("aligned_rate_all", "aligned_rate_decided", "decided_rate")


def load_counts(path: str, orders: str) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """Per eval set: sorted question ids and per-question counts.

    Returns ``{eval: {"ids": [...], "aligned", "decided", "n"[, "aligned_orig", "n_orig",
    "aligned_swapped", "n_swapped"]}}``. ``orders`` is ``pooled`` (all responses) or
    ``orig`` (original question order only).
    """
    per: Dict[str, Dict[str, Dict[str, int]]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["status"] == "unjudged":
                raise SystemExit(f"{path}: unjudged records; rescore them first")
            if orders == "orig" and rec["variant"] != "orig":
                continue
            q = per.setdefault(rec["eval"], {}).setdefault(
                rec["id"], {"aligned": 0, "decided": 0, "n": 0, "aligned_orig": 0, "n_orig": 0,
                            "aligned_swapped": 0, "n_swapped": 0})
            aligned = rec["status"] == "aligned"
            q["aligned"] += aligned
            q["decided"] += rec["status"] in ("aligned", "misaligned")
            q["n"] += 1
            q[f"aligned_{rec['variant']}"] += aligned
            q[f"n_{rec['variant']}"] += 1
    out = {}
    for ev, qs in per.items():
        ids = sorted(qs)
        out[ev] = {"ids": ids, **{k: np.array([qs[i][k] for i in ids]) for k in next(iter(qs.values()))}}
    return out


def rates(c: Dict[str, np.ndarray], idx: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Ratio-of-sums rates; with ``idx`` (resamples x questions) one value per resample."""
    def total(k):
        return c[k].sum() if idx is None else c[k][idx].sum(axis=-1)

    a, d, n = total("aligned"), total("decided"), total("n")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {"aligned_rate_all": a / n, "aligned_rate_decided": a / d, "decided_rate": d / n}
        if c["n_orig"].sum() and c["n_swapped"].sum():
            out["position_gap"] = total("aligned_orig") / total("n_orig") - total("aligned_swapped") / total("n_swapped")
    return out


def _ci(samples: np.ndarray, level: float) -> List[float]:
    lo, hi = np.nanpercentile(samples, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return [float(lo), float(hi)]


def _contrast_entry(point: float, samples: np.ndarray, level: float) -> Dict[str, Any]:
    ci = _ci(samples, level)
    return {"delta": point, "ci": ci, "lower_bound_above_zero": ci[0] > 0, "upper_bound_below_zero": ci[1] < 0}


def parse_contrast(spec: str) -> Tuple[str, str, str]:
    """``EVAL:A,B`` -> (eval, A, B). Labels may contain hyphens, so the separator is a comma."""
    ev, sep, pair = spec.partition(":")
    a, sep2, b = pair.partition(",")
    if not (sep and sep2 and ev and a and b) or "," in b:
        raise SystemExit(f"--contrast expects NAME=EVAL:A,B, got {spec!r}")
    return ev, a, b


def analyze(
    runs: Dict[str, str],
    baseline: str,
    gates: Dict[str, str],
    orders: str = "pooled",
    resamples: int = 10_000,
    seed: int = 0,
    level: float = 0.95,
    contrasts: Optional[Dict[str, str]] = None,
    differences: Optional[Dict[str, str]] = None,
    gate_contrasts: Sequence[str] = (),
) -> Dict[str, Any]:
    contrasts = {name: parse_contrast(spec) for name, spec in (contrasts or {}).items()}
    differences = dict(differences or {})
    for name, (ev, a, b) in contrasts.items():
        for label in (a, b):
            if label not in runs:
                raise SystemExit(f"--contrast {name}: {label!r} is not one of the labels")
    for name, spec in differences.items():
        c1, sep, c2 = spec.partition(",")
        if not sep or c1 not in contrasts or c2 not in contrasts:
            raise SystemExit(f"--difference {name}={spec}: expects two named contrasts C1,C2")
        if contrasts[c1][0] != contrasts[c2][0]:
            raise SystemExit(f"--difference {name}: {c1} and {c2} are on different eval sets")
        differences[name] = (c1, c2)
    for name in gate_contrasts:
        if name not in contrasts:
            raise SystemExit(f"--gate-contrast {name!r} is not a named --contrast")
    if baseline not in runs:
        raise SystemExit(f"--baseline {baseline!r} is not one of the labels {sorted(runs)}")
    for ev, arm in gates.items():
        if arm not in runs:
            raise SystemExit(f"--gate {ev}={arm}: {arm!r} is not one of the labels")
    counts = {label: load_counts(path, orders) for label, path in runs.items()}
    evals = sorted(next(iter(counts.values())))
    for label, c in counts.items():
        if sorted(c) != evals:
            raise SystemExit(f"run {label!r} covers eval sets {sorted(c)}, others cover {evals}")
        for ev in evals:
            if c[ev]["ids"] != counts[baseline][ev]["ids"]:
                raise SystemExit(f"run {label!r} and {baseline!r} differ in their {ev} questions")
    for ev in gates:
        if ev not in evals:
            raise SystemExit(f"--gate eval set {ev!r} not in the runs ({evals})")
    for name, (ev, _, _) in contrasts.items():
        if ev not in evals:
            raise SystemExit(f"--contrast {name}: eval set {ev!r} not in the runs ({evals})")

    rng = np.random.default_rng(seed)
    result: Dict[str, Any] = {
        "config": {"runs": runs, "baseline": baseline, "gates": gates, "orders": orders,
                   "resamples": resamples, "seed": seed, "level": level, "ci": "percentile, question-clustered"},
        "by_eval": {}, "gate": {},
    }
    if contrasts:
        result["config"].update({
            "contrasts": {n: f"{ev}:{a},{b}" for n, (ev, a, b) in contrasts.items()},
            "differences": {n: f"{c1},{c2}" for n, (c1, c2) in differences.items()},
            "gate_contrasts": list(gate_contrasts),
        })
    boots: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for ev in evals:  # sorted, so the RNG stream per eval set is fixed
        n_q = len(counts[baseline][ev]["ids"])
        idx = rng.integers(0, n_q, size=(resamples, n_q))
        boot = {label: rates(c[ev], idx) for label, c in counts.items()}
        boots[ev] = boot
        entry = {"n_questions": n_q, "adapters": {}}
        for label, c in counts.items():
            point = rates(c[ev])
            entry["adapters"][label] = {
                "n_responses": int(c[ev]["n"].sum()),
                **{m: {"value": float(point[m]), "ci": _ci(boot[label][m], level)} for m in point},
            }
        result["by_eval"][ev] = entry
        if ev in gates:
            arm = gates[ev]
            contrast = {}
            for m in ("aligned_rate_all", "aligned_rate_decided"):
                delta = float(rates(counts[arm][ev])[m] - rates(counts[baseline][ev])[m])
                ci = _ci(boot[arm][m] - boot[baseline][m], level)
                contrast[m] = {"delta": delta, "ci": ci, "lower_bound_above_zero": ci[0] > 0}
            result["gate"][ev] = {"arm": arm, "baseline": baseline, **contrast}
    # Named contrasts and differences reuse each eval set's resample matrix, so they are paired.
    if contrasts:
        result["contrasts"], result["differences"] = {}, {}
        point = {(lab, ev): rates(counts[lab][ev]) for lab in runs for ev in evals}
        cpoint, csamp = {}, {}
        for name, (ev, a, b) in contrasts.items():
            entry = {"eval": ev, "a": a, "b": b, "gate": name in gate_contrasts}
            for m in ("aligned_rate_all", "aligned_rate_decided"):
                cpoint[name, m] = float(point[a, ev][m] - point[b, ev][m])
                csamp[name, m] = boots[ev][a][m] - boots[ev][b][m]
                entry[m] = _contrast_entry(cpoint[name, m], csamp[name, m], level)
            result["contrasts"][name] = entry
        for name, (c1, c2) in differences.items():
            entry = {"eval": contrasts[c1][0], "of": [c1, c2]}
            for m in ("aligned_rate_all", "aligned_rate_decided"):
                entry[m] = _contrast_entry(cpoint[c1, m] - cpoint[c2, m], csamp[c1, m] - csamp[c2, m], level)
            result["differences"][name] = entry
    if gates or gate_contrasts:
        verdicts = [g["aligned_rate_all"]["lower_bound_above_zero"] for g in result["gate"].values()]
        verdicts += [result["contrasts"][n]["aligned_rate_all"]["lower_bound_above_zero"] for n in gate_contrasts]
        result["gate_passes"] = all(verdicts)
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", nargs="+", action="extend", required=True, help="adapter labels")
    p.add_argument("--responses", nargs="+", action="extend", required=True,
                   help="judged preference.jsonl per label, in the same order")
    p.add_argument("--baseline", required=True, help="label of the baseline run")
    p.add_argument("--gate", action="append", default=[], metavar="EVAL=LABEL",
                   help="gate contrast: LABEL minus baseline on eval set EVAL (repeatable)")
    p.add_argument("--contrast", action="append", default=[], metavar="NAME=EVAL:A,B",
                   help="named contrast: rate(A) - rate(B) on eval set EVAL (repeatable)")
    p.add_argument("--difference", action="append", default=[], metavar="NAME=C1,C2",
                   help="difference of two named contrasts on the same eval set (repeatable)")
    p.add_argument("--gate-contrast", action="append", default=[], metavar="NAME",
                   help="include a named contrast in the gate verdict (repeatable)")
    p.add_argument("--orders", choices=["pooled", "orig"], default="pooled",
                   help="pool both question orders (default) or use the original order only")
    p.add_argument("--resamples", type=int, default=10_000)
    p.add_argument("--level", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="path to results.json")
    return p.parse_args(argv)


def _pairs(items: Sequence[str], flag: str) -> Dict[str, str]:
    out = {}
    for it in items:
        key, sep, val = it.partition("=")
        if not sep or not key or not val:
            raise SystemExit(f"{flag} expects KEY=VALUE, got {it!r}")
        if key in out:
            raise SystemExit(f"{flag}: {key!r} given twice")
        out[key] = val
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if len(args.labels) != len(args.responses):
        raise SystemExit(f"{len(args.labels)} --labels but {len(args.responses)} --responses")
    if len(set(args.labels)) != len(args.labels):
        raise SystemExit("--labels must be unique")
    res = analyze(dict(zip(args.labels, args.responses)), args.baseline, _pairs(args.gate, "--gate"),
                  args.orders, args.resamples, args.seed, args.level,
                  contrasts=_pairs(args.contrast, "--contrast"), differences=_pairs(args.difference, "--difference"),
                  gate_contrasts=args.gate_contrast)
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    for ev, entry in res["by_eval"].items():
        print(f"{ev} ({entry['n_questions']} questions)")
        for label, a in entry["adapters"].items():
            m = a["aligned_rate_all"]
            print(f"  {label:<40} aligned_rate_all {m['value']:.3f} [{m['ci'][0]:.3f}, {m['ci'][1]:.3f}]"
                  f"  decided {a['decided_rate']['value']:.3f}")
    for ev, g in res["gate"].items():
        m = g["aligned_rate_all"]
        print(f"gate {ev}: {g['arm']} - {g['baseline']} = {m['delta']:+.3f} [{m['ci'][0]:+.3f}, {m['ci'][1]:+.3f}]"
              f" {'PASS' if m['lower_bound_above_zero'] else 'FAIL'}")
    for kind in ("contrasts", "differences"):
        for name, c in res.get(kind, {}).items():
            m = c["aligned_rate_all"]
            what = f"{c['a']} - {c['b']}" if kind == "contrasts" else f"{c['of'][0]} - {c['of'][1]}"
            flag = "above 0" if m["lower_bound_above_zero"] else "below 0" if m["upper_bound_below_zero"] else "includes 0"
            print(f"{kind[:-1]} {name} ({c['eval']}): {what} = {m['delta']:+.3f} [{m['ci'][0]:+.3f}, {m['ci'][1]:+.3f}]"
                  f" CI {flag}{' [gate]' if c.get('gate') else ''}")
    if "gate_passes" in res:
        print("GATE", "PASSES" if res["gate_passes"] else "FAILS")
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
