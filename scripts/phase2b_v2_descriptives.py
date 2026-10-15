"""Descriptive extras for Phase 2b (identity v2), computed from the committed analysis outputs.

    python scripts/phase2b_v2_descriptives.py   # writes msm/analyses/phase2b-v2-descriptives/results.json

Pre-registered in docs/preregistration/phase2b-section31-seeds.md §6 ("Match to the released
adapters and to Fig. 2", "Also reported"), items no analysis module computes:
- Spearman rank correlation, per eval, between our six arm seed means (judge, aligned_rate_all)
  and Fig. 2's six values, and between ours and the released adapters' (Phase 1 test relabel).
- aligned_rate_decided versions of the gate contrasts: per-seed differences of the per-run
  decided-only rates from the question-level analysis, their mean, and a seed-only Student-t
  95% interval (3 df; no question-sampling term, so narrower than the primary CIs).
- P5: how many of the 12 released arm x eval cells lie inside our per-arm prediction intervals.

Reads only committed files under msm/analyses/, so it is run directly rather than through the
launcher. Run from the repo root.
"""
import json
import os
import statistics

SEEDS_JUDGE = "msm/analyses/phase2b-v2-analysis-seeds-judge/results.json"
RUNS = "msm/analyses/phase2b-v2-analysis-runs-test/results.json"
OUT = "msm/analyses/phase2b-v2-descriptives/results.json"
EVALS = ("affordability", "america")
# Fig. 2 value labels, arm order baseline, AFT, MSM(aff), MSM(Am), MSM+AFT(aff), MSM+AFT(Am)
# (scripts/plot_fig2_comparison.py) mapped to our arm codes.
ARM_ORDER = ("A4", "A3", "A6", "A5", "A2", "A1")
FIG2 = {"affordability": [0.23, 0.32, 0.38, 0.28, 0.48, 0.29], "america": [0.38, 0.36, 0.36, 0.52, 0.38, 0.55]}
GATE = {"H1": ("affordability", "A2", "A1"), "H2": ("america", "A1", "A2"),
        "G1": ("affordability", "A2", "A4"), "G2": ("america", "A1", "A4")}
T_975_3DF = 3.182


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return num / (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5


def main():
    seeds = json.load(open(SEEDS_JUDGE))
    runs = json.load(open(RUNS))
    out = {"inputs": [SEEDS_JUDGE, RUNS], "spearman": {}, "decided_gate": {}, "p5": {}}
    inside = 0
    for ev in EVALS:
        ours = [seeds["by_eval"][ev]["arms"][a]["mean"] for a in ARM_ORDER]
        rel = [seeds["within"][f"{a}_{ev}_released"]["reference_value"] for a in ARM_ORDER]
        out["spearman"][ev] = {"arms": ARM_ORDER, "ours": ours, "fig2": FIG2[ev], "released": rel,
                               "ours_vs_fig2": spearman(ours, FIG2[ev]), "ours_vs_released": spearman(ours, rel)}
        cells = {a: seeds["within"][f"{a}_{ev}_released"]["in_prediction_interval"] for a in ARM_ORDER}
        out["p5"][ev] = cells
        inside += sum(cells.values())
    out["p5"]["inside_of_12"] = inside
    rate = {ev: runs["by_eval"][ev]["adapters"] for ev in EVALS}
    for name, (ev, a, b) in GATE.items():
        per = [rate[ev][f"ours-{a}-s{s}"]["aligned_rate_decided"]["value"]
               - rate[ev][f"ours-{b}-s{s}"]["aligned_rate_decided"]["value"] for s in range(4)]
        m, sd = statistics.mean(per), statistics.stdev(per)
        half = T_975_3DF * sd / 2
        out["decided_gate"][name] = {"eval": ev, "a": a, "b": b, "per_seed": per, "mean": m,
                                     "ci_seed_only": [m - half, m + half]}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    for ev in EVALS:
        s = out["spearman"][ev]
        print(f"spearman {ev}: ours vs Fig. 2 {s['ours_vs_fig2']:+.3f}, ours vs released {s['ours_vs_released']:+.3f}")
    for name, g in out["decided_gate"].items():
        print(f"decided-only {name} ({g['eval']}): {g['mean']:+.3f} [{g['ci_seed_only'][0]:+.3f}, "
              f"{g['ci_seed_only'][1]:+.3f}] per seed {' '.join(f'{v:+.3f}' for v in g['per_seed'])}")
    print(f"P5: {inside}/12 released cells inside our prediction intervals; {out['p5']}")


if __name__ == "__main__":
    main()
