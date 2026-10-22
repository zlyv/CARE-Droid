#!/usr/bin/env bash
# Start the full resumable P31 and new-2017 Stage-II asset extraction in tmux.
set -euo pipefail
PROJECT=/home/boyuanjie/projects/tasa-droid
OUT="$PROJECT/outputs/care_expanded2017_benign_aware_v2"
SCRIPT="$PROJECT/scripts/care-expanded2017-benign-aware-v2/resume.sh"
SESSION=care-expanded2017-v2-extract
test -f "$OUT/dataset_audit.json"
test -f "$OUT/formal_p31_replay.json"
test -f "$OUT/stage2_smoke/builder_status.json"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session already exists: $SESSION" >&2
  exit 2
fi
tmux new-session -d -s "$SESSION" \
  "cd '$PROJECT' && bash '$SCRIPT' > '$OUT/extraction.console.log' 2>&1"
echo "Started $SESSION; monitor: tail -F $OUT/extraction.console.log"
