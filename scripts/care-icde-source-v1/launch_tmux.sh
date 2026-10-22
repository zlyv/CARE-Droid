#!/usr/bin/env bash
set -euo pipefail

PROJECT="${ICDE_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SESSION="${ICDE_SESSION:-care-icde-stage-a-v1}"

version=1
while true; do
  OUT="$PROJECT/outputs/care-icde-source-v${version}"
  CONSOLE="$PROJECT/outputs/care-icde-source-v${version}.console.log"
  if [[ ! -e "$OUT" && ! -e "$CONSOLE" ]]; then break; fi
  version=$((version + 1))
done

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session exists: $SESSION"; exit 1
fi
mkdir -p "$PROJECT/outputs"
tmux new-session -d -s "$SESSION" \
  "cd '$PROJECT' && ICDE_OUTPUT='$OUT' bash '$SCRIPT_DIR/run_stage_a.sh' 2>&1 | tee -a '$CONSOLE'"
echo "Started: $SESSION"
echo "Stage: A only (read-only feasibility audit; no APK execution)"
echo "Output: $OUT"
echo "Monitor: tail -F $CONSOLE"
echo "Attach: tmux attach -t $SESSION"

