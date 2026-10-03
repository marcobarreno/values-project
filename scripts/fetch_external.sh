#!/usr/bin/env bash
# Clone the third-party repos we depend on into external/, checked out at the
# exact commits we use (submodules follow the superproject's pins).
#   bash scripts/fetch_external.sh
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"

fetch() {  # fetch <url> <dir-name> <commit>
  local url="$1" dest="$root/external/$2" rev="$3"
  [ -d "$dest/.git" ] || git clone "$url" "$dest"
  git -C "$dest" cat-file -e "$rev^{commit}" 2>/dev/null || git -C "$dest" fetch origin
  git -C "$dest" checkout --quiet "$rev"
  git -C "$dest" submodule update --init --recursive
  echo "$2 @ $(git -C "$dest" rev-parse --short HEAD)"
}

# Model Spec Midtraining (Li et al., arXiv 2605.02087): data-gen pipelines, specs, AM eval.
fetch https://github.com/chloeli-15/model_spec_midtraining model_spec_midtraining \
  e8288a84912ba32af68ad15f2e52a7c1b4e81891
