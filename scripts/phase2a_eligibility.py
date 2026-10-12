"""Phase 2a dev eligibility check (docs/preregistration/phase2a-section31-training.md §5).

    python scripts/phase2a_eligibility.py RUN_DIR [RUN_DIR ...]

For every run x eval set: decided rate >= 0.70, order agreement >= 0.95, and judge errors +
invalid verdicts <= 1%. The last is checked conservatively with the share of `unparsed`
labels, which counts judge errors, invalid verdicts and empty responses. Exits 1 if any
cell fails.
"""
import json
import sys

ok = True
for run in sys.argv[1:]:
    s = json.load(open(f"{run}/summary.json"))
    for ev, e in sorted(s["by_eval"].items()):
        unparsed = e["counts"]["unparsed"] / e["n_responses"]
        cell = e["decided_rate"] >= 0.70 and e["order_agreement_rate"] >= 0.95 and unparsed <= 0.01
        ok &= cell
        print(f"{run:<50} {ev:<14} decided {e['decided_rate']:.3f}  order agreement {e['order_agreement_rate']:.3f}"
              f"  unparsed {unparsed:.3f}  length stops {e['length_stop_rate']:.3f}  label methods {e['label_methods']}"
              f"  {'OK' if cell else 'FAIL'}")
print("ELIGIBLE" if ok else "NOT ELIGIBLE")
sys.exit(0 if ok else 1)
