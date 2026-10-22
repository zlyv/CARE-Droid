#!/usr/bin/env bash
# Read-only HDBC-V1 status helper.
set -euo pipefail
ROOT=/home/boyuanjie/projects/tasa-droid
OUT="${HDBC_OUTPUT:-$ROOT/outputs/care_hdbc_v1}"
for name in launch_info.txt phase_status.json run_state.json exit_code.txt; do
  if [[ -f "$OUT/$name" ]]; then echo "===== $name ====="; cat "$OUT/$name"; fi
done
if [[ -f "$OUT/run.log" ]]; then echo '===== recent log ====='; tail -n 30 "$OUT/run.log"; fi
