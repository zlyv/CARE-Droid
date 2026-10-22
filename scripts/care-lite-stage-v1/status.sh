#!/usr/bin/env bash
set -euo pipefail
OUT=/home/boyuanjie/projects/tasa-droid/outputs/care_lite_stage_v1
printf '===== STATE =====\n'
if [[ -f "$OUT/run_state.json" ]]; then cat "$OUT/run_state.json"; else echo NOT_STARTED; fi
printf '\n===== PHASES =====\n'
if [[ -f "$OUT/phase_status.json" ]]; then cat "$OUT/phase_status.json"; else echo NOT_STARTED; fi
printf '\n===== LOG =====\n'
if [[ -f "$OUT/run.log" ]]; then tail -n 25 "$OUT/run.log"; fi
