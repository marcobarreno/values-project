#!/bin/bash
# Build the Phase 4 AM-eval venv (msm/.venv-am) from the hashed lock. Run from the repo root.
#   bash scripts/setup_am_env.sh
# To change a version: edit msm/env/requirements-am.in, then re-run with LOCK=1 to recompile the lock.
set -euo pipefail
cd "$(dirname "$0")/.."
IN=msm/env/requirements-am.in
LOCK_FILE=msm/env/requirements-am.lock
if [ "${LOCK:-0}" = 1 ]; then
  uv pip compile "$IN" --python-version 3.12 --python-platform x86_64-manylinux_2_28 \
    --index-url https://pypi.org/simple --generate-hashes --emit-index-url -o "$LOCK_FILE"
fi
[ -x msm/.venv-am/bin/python ] || uv venv --python 3.12 msm/.venv-am
uv pip sync --python msm/.venv-am/bin/python --require-hashes "$LOCK_FILE"
msm/.venv-am/bin/python - <<'PY'
import torch, vllm, inspect_ai, anthropic
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("vllm", vllm.__version__, "inspect_ai", inspect_ai.__version__, "anthropic", anthropic.__version__)
PY
