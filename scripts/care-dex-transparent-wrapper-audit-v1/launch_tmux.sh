#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${TW_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${TW_OUTPUT:-$PROJECT/outputs/care-dex-transparent-wrapper-audit-v1}"
SESSION="${TW_SESSION:-care-dex-wrapper-audit-v1}"
CONSOLE="$PROJECT/outputs/care-dex-transparent-wrapper-audit-v1.console.log"

mkdir -p "$PROJECT/outputs" "$OUTPUT/logs"
touch "$CONSOLE"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session already exists: $SESSION"
  exit 1
fi

tmux new-session -d -s "$SESSION" \
  "cd '$PROJECT' && TW_PROJECT='$PROJECT' TW_OUTPUT='$OUTPUT' TW_PYTHON='${TW_PYTHON:-}' TW_FORMAL='${TW_FORMAL:-}' TW_ENDPOINT_CACHE='${TW_ENDPOINT_CACHE:-}' TW_ATTENTION='${TW_ATTENTION:-}' TW_MAPPING='${TW_MAPPING:-}' TW_APK_INDEX='${TW_APK_INDEX:-}' TW_APK_ROOTS='${TW_APK_ROOTS:-}' TW_APK_TIMEOUT_SECONDS='${TW_APK_TIMEOUT_SECONDS:-900}' bash '$SCRIPT_DIR/run_on_server.sh' 2>&1 | tee -a '$CONSOLE'"

echo "Started: $SESSION"
echo "Output: $OUTPUT"
echo "Monitor: tail -F $CONSOLE"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"

