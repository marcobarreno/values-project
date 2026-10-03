#!/bin/bash
# Download MSM artifacts from HF. Usage:
#   bash models/download.sh datasets            # all 10 datasets (~0.5 GB) -> data/hf/
#   bash models/download.sh cheese              # 6 Llama-8B adapters for the pro-America/pro-affordability result (~4.2 GB)
#   bash models/download.sh single-value        # 19 Llama-8B adapters for the 6-value result (~13 GB)
#   bash models/download.sh philosophy|general|rules|value-aug|rules-aug|scaling   # Qwen 14B/32B adapters (26-125 GB each!)
#   bash models/download.sh repo chloeli/<name> # one repo
# Full list with sizes: models/hf_manifest.json (fields: collection, type, repo_id, GB, is_adapter, gated, private)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HF="$ROOT/.venv/bin/hf"
MANIFEST="$ROOT/models/hf_manifest.json"
export HF_HUB_ENABLE_HF_TRANSFER=0

dl() {  # dl <type> <repo_id>
  local typ="$1" rid="$2" name="${2#chloeli/}"
  if [ "$typ" = dataset ]; then
    "$HF" download --repo-type dataset "$rid" --local-dir "$ROOT/data/hf/$name"
  else
    "$HF" download "$rid" --local-dir "$ROOT/models/$name"
  fi
}

select_rows() {  # select_rows <python filter expr over (col,typ,rid)>
  python3 - "$MANIFEST" "$1" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1])); expr=sys.argv[2]; seen=set()
for col,typ,rid,gb,adapter,gated,private in rows:
    if rid in seen: continue
    if eval(expr): seen.add(rid); print(typ, rid)
PY
}

case "${1:-}" in
  datasets)     sel="typ=='dataset'";;
  cheese)       sel="typ=='model' and 'Pro-Americ' in col";;
  single-value) sel="typ=='model' and 'Single' in col";;
  philosophy)   sel="typ=='model' and 'Philosophy' in col";;
  general)      sel="typ=='model' and 'General' in col";;
  rules)        sel="typ=='model' and col.endswith('Rules Spec')";;
  value-aug)    sel="typ=='model' and 'Value-Aug' in col";;
  rules-aug)    sel="typ=='model' and 'Rules-Aug' in col";;
  scaling)      sel="typ=='model' and 'Scaling' in col";;
  repo)         dl model "$2"; exit 0;;
  *) sed -n '2,8p' "$0"; exit 1;;
esac
select_rows "$sel" | while read -r typ rid; do echo ">>> $rid"; dl "$typ" "$rid"; done
echo "DONE $1"
