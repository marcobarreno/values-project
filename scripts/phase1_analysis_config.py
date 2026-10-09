"""Write a Phase 1 `analyze` config that pins the six adapters' run directories by hash.

    python scripts/phase1_analysis_config.py TAG ORDERS PATTERN [ARM=RUN_DIR ...]

TAG names the cell (config configs/phase1/analysis-TAG-ORDERS.yaml, output
msm/analyses/phase1-analysis-TAG-ORDERS). ORDERS is pooled or orig. PATTERN is a
run directory name with {arm}, e.g. phase1-test-relabel-{arm}. ARM=RUN_DIR
overrides the directory for one adapter (for cells whose runs come from two
families). Run from the repo root; the run directories must exist locally.
"""
import os
import sys

sys.path.insert(0, "src")
from msm_repro.launch import sha256_path  # noqa: E402

ARMS = ["baseline", "cheese-aft", "pro-america-spec-msm", "pro-america-spec-msm-cheese-aft",
        "pro-affordability-spec-msm", "pro-affordability-spec-msm-cheese-aft"]

tag, orders, pattern, *overrides = sys.argv[1:]
assert orders in ("pooled", "orig"), orders
dirs = {a: pattern.format(arm=a) for a in ARMS}
for o in overrides:
    arm, _, d = o.partition("=")
    assert arm in dirs, arm
    dirs[arm] = d
for d in dirs.values():
    assert os.path.isdir(f"msm/runs/{d}"), f"missing msm/runs/{d}"
kind = "TEST split: the pre-registered primary analysis (gate verdict)" if tag.startswith("test") else "dev-sweep cell (robustness analysis)"
name = f"phase1-analysis-{tag}-{orders}"
files = "\n".join(f"  {a}: {{path: msm/runs/{d}, sha256: {sha256_path(f'msm/runs/{d}')}}}" for a, d in dirs.items())
path = f"configs/phase1/analysis-{tag}-{orders}.yaml"
open(path, "w").write(f"""name: {name}
command: analyze
description: >
  Phase 1 analysis, {kind}: runs {pattern.format(arm="*")}{" (with overrides)" if overrides else ""}, question orders {orders}. Question-clustered
  paired bootstrap (10,000 resamples, seed 0) and the two gate contrasts, as pre-registered in
  docs/preregistration/phase1-section31.md §6. Output committed under msm/analyses/.
out_dir: msm/analyses/{name}
files:
{files}
args:
  labels:
{chr(10).join(f"    - {a}" for a in ARMS)}
  responses:
{chr(10).join(f"    - file:{a}/preference.jsonl" for a in ARMS)}
  baseline: baseline
  gate:
    - affordability=pro-affordability-spec-msm-cheese-aft
    - america=pro-america-spec-msm-cheese-aft
  orders: {orders}
  resamples: 10000
  seed: 0
""")
print(path)
