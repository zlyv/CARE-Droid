#!/usr/bin/env bash
set -euo pipefail
PROJECT="${HCB_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT="$PROJECT/scripts/care-host-control-binding-observability-v1"
OUTPUT="$PROJECT/outputs/care-host-control-binding-observability-v1"
SESSION="${HCB_SESSION:-care-host-control-binding-v1}"
if [ -e "$OUTPUT" ] || [ -e "$OUTPUT.console.log" ]; then
  echo "Experiment output/log exists: refusing overwrite. Inspect it or use a new version."; exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session already exists: $SESSION"; exit 1
fi
mkdir -p "$PROJECT/outputs"
env_args=("HCB_PROJECT=$PROJECT" "HCB_PYTHON=${HCB_PYTHON:-/home/boyuanjie/envs/tif/bin/python}")
for name in HCB_PARENT HCB_ENDPOINT HCB_C43_CACHE HCB_ROLE_MANIFEST; do
  if [ -n "${!name:-}" ]; then env_args+=("$name=${!name}"); fi
done
printf -v invocation '%q ' env "${env_args[@]}" bash "$SCRIPT/run_on_server.sh"
printf -v project_q '%q' "$PROJECT"
printf -v console_q '%q' "$OUTPUT.console.log"
tmux new-session -d -s "$SESSION" "cd $project_q && $invocation >> $console_q 2>&1"
echo "Started: $SESSION"
echo "Monitor: tail -F $OUTPUT.console.log"
echo "CPU only; reuses 799 existing scans; no new DEX extraction or model training"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"
