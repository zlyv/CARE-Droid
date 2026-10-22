#!/usr/bin/env bash
set -euo pipefail
PROJECT="${LCC_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT="$PROJECT/scripts/care-local-control-constraint-feasibility-v1"
OUTPUT="${LCC_OUTPUT:-$PROJECT/outputs/care-local-control-constraint-feasibility-v1}"
SESSION="${LCC_SESSION:-care-lcc-feasibility-v1}"
CONSOLE="$OUTPUT.console.log"
mkdir -p "$PROJECT/outputs" "$OUTPUT/logs"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session already exists: $SESSION"; exit 1
fi
env_args=("LCC_PROJECT=$PROJECT" "LCC_OUTPUT=$OUTPUT" "LCC_WORKERS=${LCC_WORKERS:-4}" "LCC_PYTHON=${LCC_PYTHON:-/home/boyuanjie/envs/tif/bin/python}" "LCC_APK_TIMEOUT_SECONDS=${LCC_APK_TIMEOUT_SECONDS:-1200}")
for name in LCC_APK_INDEX LCC_APK_ROOTS LCC_ENDPOINT LCC_FORMAL_P2_ROOT; do
  if [ -n "${!name:-}" ]; then env_args+=("$name=${!name}"); fi
done
printf -v invocation '%q ' env "${env_args[@]}" bash "$SCRIPT/run_on_server.sh"
printf -v project_q '%q' "$PROJECT"
printf -v console_q '%q' "$CONSOLE"
tmux new-session -d -s "$SESSION" "cd $project_q && $invocation >> $console_q 2>&1"
echo "Started: $SESSION"
echo "Monitor: tail -F $CONSOLE"
echo "Attach: tmux attach -t $SESSION"
echo "Workers: ${LCC_WORKERS:-4}; GPU not required"
