#!/usr/bin/env bash
# Run launcher configs one after another, for unattended work.
#
#   bash scripts/run_queue.sh QUEUE_DIR
#
# QUEUE_DIR/queue.txt lists config paths (repo-relative), one per line. Lines may be
# appended while the queue runs: the next config is read after each run finishes.
# Finished configs are appended to QUEUE_DIR/done.txt, and each launch's output goes
# to QUEUE_DIR/<config name>.log.
#
# Exit handling:
# - exit 0: the run succeeded; go on to the next config.
# - exit 2 where every "cannot launch" problem is uncommitted changes in src/ or the
#   config (someone is mid-commit): retry every 60 s, for at most RETRIES minutes.
# - anything else stops the queue with that exit code: any other launcher refusal
#   (untracked config, existing output directory, hash mismatch), a config error, a
#   failed run, or exit 3 from eval_preference/rescore (judge API failure; see
#   src/msm_repro/DESIGN.md §10). Later configs are not attempted.
set -u
Q=${1:?usage: run_queue.sh QUEUE_DIR}
RETRIES=${RETRIES:-30}
cd "$(dirname "$0")/.."
touch "$Q/done.txt"
while true; do
  next=$(grep -vxF -f "$Q/done.txt" "$Q/queue.txt" | grep -v '^\s*$' | head -1)
  if [ -z "$next" ]; then echo "queue empty $(date -Iseconds)"; exit 0; fi
  log="$Q/$(basename "$next" .yaml).log"
  for try in $(seq 1 "$RETRIES"); do
    echo "=== $next try $try $(date -Iseconds)"
    PYTHONPATH=src msm/.venv/bin/python -m msm_repro.launch "$next" > "$log" 2>&1
    rc=$?
    problems=$(grep "^cannot launch:" "$log")
    if [ "$rc" -eq 2 ] && [ -n "$problems" ] && ! grep -v "uncommitted changes" <<< "$problems" | grep -q .; then
      echo "  dirty tree; retrying in 60 s"; sleep 60; continue
    fi
    break
  done
  echo "exit $rc $(date -Iseconds) $next"
  if [ "$rc" -ne 0 ]; then
    echo "stopping the queue: $next exited $rc (see $log)"; tail -5 "$log"; exit "$rc"
  fi
  echo "$next" >> "$Q/done.txt"
done
