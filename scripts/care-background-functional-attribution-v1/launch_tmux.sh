#!/usr/bin/env bash
# BFA-V1 safe tmux launcher; refuses to reuse an existing session or console file.
set -euo pipefail
ROOT=/home/boyuanjie/projects/tasa-droid
OUT="$ROOT/outputs/care_background_functional_attribution_v1"
SESSION=care-bfa-v1
CONSOLE="$OUT/console.log"
test -f "$OUT/intervention_smoke.json" || { echo 'BFA smoke test required'; exit 1; }
test ! -e "$CONSOLE" || { echo 'BFA console already exists; refusing overwrite'; exit 1; }
if tmux has-session -t "$SESSION" 2>/dev/null; then echo 'BFA session already exists'; exit 1; fi
tmux new-session -d -s "$SESSION" "cd '$ROOT' && bash scripts/care-background-functional-attribution-v1/pipeline.sh > '$CONSOLE' 2>&1"
echo "Started: $SESSION"
echo "Monitor: tail -F $CONSOLE"
echo "Attach: tmux attach -t $SESSION"
