#!/usr/bin/env bash
# RCDB-V1 status reader; read-only inspection of the chosen output directory.
set -Eeuo pipefail
: "${1:?pass the absolute RCDB output directory}"
OUT="$1"
for f in run_state.json phase_status.json exit_code.txt final_decision.json; do
  if [ -f "$OUT/$f" ]; then echo "===== $f ====="; cat "$OUT/$f"; fi
done
if [ -f "$OUT/completed.flag" ]; then echo COMPLETED_FLAG_PRESENT; fi
