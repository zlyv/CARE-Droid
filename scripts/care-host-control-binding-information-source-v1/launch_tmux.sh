#!/usr/bin/env bash
set -euo pipefail
PROJECT="${HBI_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT="$PROJECT/scripts/care-host-control-binding-information-source-v1"
OUTPUT="$PROJECT/outputs/care-host-control-binding-information-source-v1"
SESSION="${HBI_SESSION:-care-host-binding-information-v1}"
if [ -e "$OUTPUT" ] || [ -e "$OUTPUT.console.log" ]; then echo "Output/log exists; refusing overwrite."; exit 1; fi
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "Session exists: $SESSION"; exit 1; fi
mkdir -p "$PROJECT/outputs"
env_args=("HBI_PROJECT=$PROJECT" "HBI_PYTHON=${HBI_PYTHON:-/home/boyuanjie/envs/tif/bin/python}")
if [ -n "${HBI_OBSERVABILITY:-}" ]; then env_args+=("HBI_OBSERVABILITY=$HBI_OBSERVABILITY"); fi
printf -v invocation '%q ' env "${env_args[@]}" bash "$SCRIPT/run_on_server.sh"
printf -v project_q '%q' "$PROJECT"
printf -v console_q '%q' "$OUTPUT.console.log"
tmux new-session -d -s "$SESSION" "cd $project_q && $invocation >> $console_q 2>&1"
echo "Started: $SESSION"
echo "Monitor: tail -F $OUTPUT.console.log"
echo "CPU only; no new APK scans or C43 retraining; fixed auxiliary readouts"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"
