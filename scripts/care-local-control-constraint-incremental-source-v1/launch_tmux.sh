#!/usr/bin/env bash
set -euo pipefail
PROJECT="${LCIP_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${LCIP_OUTPUT:-$PROJECT/outputs/care-local-control-constraint-incremental-source-v1}"
SESSION="${LCIP_SESSION:-care-lcc-incremental-source-v1}"
SCRIPT="$PROJECT/scripts/care-local-control-constraint-incremental-source-v1"
if [ "$OUTPUT" != "$PROJECT/outputs/care-local-control-constraint-incremental-source-v1" ]; then
  echo "Refusing non-experiment output directory: $OUTPUT"; exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "Session exists: $SESSION"; exit 1; fi
mkdir -p "$OUTPUT/logs"
env_args=("LCIP_PROJECT=$PROJECT" "LCIP_OUTPUT=$OUTPUT" "LCIP_WORKERS=${LCIP_WORKERS:-4}" "LCIP_PYTHON=${LCIP_PYTHON:-/home/boyuanjie/envs/tif/bin/python}")
for name in LCIP_APK_INDEX LCIP_APK_ROOTS LCIP_ENDPOINT LCIP_LCC_V2_OUTPUT LCIP_APK_TIMEOUT_SECONDS; do
  if [ -n "${!name:-}" ]; then env_args+=("$name=${!name}"); fi
done
printf -v invocation '%q ' env "${env_args[@]}" bash "$SCRIPT/run_on_server.sh"
printf -v project_q '%q' "$PROJECT"
printf -v console_q '%q' "$OUTPUT.console.log"
tmux new-session -d -s "$SESSION" "cd $project_q && $invocation >> $console_q 2>&1"
echo "Started: $SESSION"
echo "Monitor: tail -F $OUTPUT.console.log"
echo "CPU workers: ${LCIP_WORKERS:-4}; GPU not required"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"
