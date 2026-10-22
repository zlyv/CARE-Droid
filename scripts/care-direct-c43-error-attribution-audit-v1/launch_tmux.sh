#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${ECA_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${ECA_OUTPUT:-$PROJECT/outputs/care-direct-c43-error-attribution-audit-v1}"
SESSION="care-c43-error-attribution-v1"
command -v tmux >/dev/null || { echo "tmux is not installed"; exit 1; }
[[ ! -e "$OUTPUT/final_decision.json" ]] || { echo "Audit already finished; will not overwrite"; exit 1; }
case "$(realpath -m "$OUTPUT")" in
  "$(realpath "$PROJECT")"/outputs/care-direct-c43-error-attribution-audit-v*) ;;
  *) echo "BLOCKED_BY_CONTRACT: unsafe output directory"; exit 1 ;;
esac
mkdir -p "$PROJECT/outputs"
tmux new-session -d -s "$SESSION" "cd '$PROJECT' && bash '$SCRIPT_DIR/run_on_server.sh' 2>&1 | tee '$OUTPUT.console.log'"
echo "Started: $SESSION"
echo "Monitor: tail -F $OUTPUT.console.log"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b then d"
