#!/usr/bin/env bash
set -euo pipefail
: "${1:?Usage: bash status.sh /home/boyuanjie/projects/tasa-droid/outputs/care_lfc_v1/run_TIMESTAMP}"
RUN="$1"
printf '===== run_state =====\n'
test -f "$RUN/run_state.json" && cat "$RUN/run_state.json" || true
printf '===== phase_status =====\n'
test -f "$RUN/phase_status.json" && cat "$RUN/phase_status.json" || true
printf '===== log tail =====\n'
test -f "$RUN/run.log" && tail -n 25 "$RUN/run.log" || true
