#!/bin/bash
# Build the msm_repro GPU venv (msm/.venv) from the hashed lock. Run from the repo root.
#   bash scripts/setup_gpu_env.sh
# To change a version: edit msm/env/requirements-gpu.in, then re-run with LOCK=1 to recompile the lock.
set -euo pipefail
cd "$(dirname "$0")/.."
IN=msm/env/requirements-gpu.in
LOCK_FILE=msm/env/requirements-gpu.lock
if [ "${LOCK:-0}" = 1 ]; then
  uv pip compile "$IN" --python-version 3.12 --python-platform x86_64-manylinux_2_28 \
    --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 \
    --index-strategy unsafe-best-match --generate-hashes --emit-index-url -o "$LOCK_FILE"
fi
[ -x msm/.venv/bin/python ] || uv venv --python 3.12 msm/.venv
uv pip sync --python msm/.venv/bin/python --require-hashes --index-strategy unsafe-best-match "$LOCK_FILE"
msm/.venv/bin/python - <<'PY'
import torch, transformers, peft, trl
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__, "peft", peft.__version__, "trl", trl.__version__)
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
