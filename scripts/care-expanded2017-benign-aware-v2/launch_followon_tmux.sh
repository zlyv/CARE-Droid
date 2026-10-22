#!/usr/bin/env bash
# Launch a detached watcher; it will not train until the extraction gate passes.
set -euo pipefail
PROJECT=/home/boyuanjie/projects/tasa-droid
OUT="$PROJECT/outputs/care_expanded2017_benign_aware_v2"
SCRIPT="$PROJECT/scripts/care-expanded2017-benign-aware-v2/followon.sh"
SESSION=care-expanded2017-v2-followon
test -f "$OUT/dataset_audit.json"
test -f "$OUT/formal_p31_replay.json"
test -f "$SCRIPT"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session already exists: $SESSION" >&2
  exit 2
fi
tmux new-session -d -s "$SESSION" \
  "cd '$PROJECT' && bash '$SCRIPT' > '$OUT/followon.console.log' 2>&1"
echo "Started $SESSION; monitor: tail -F $OUT/followon.console.log"
