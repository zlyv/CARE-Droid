#!/usr/bin/env bash
# Start only the isolated smoke test in tmux; no training or broad extraction.
set -euo pipefail
PROJECT=/home/boyuanjie/projects/tasa-droid
OUT="$PROJECT/outputs/care_expanded2017_benign_aware_v2"
SCRIPT="$PROJECT/scripts/care-expanded2017-benign-aware-v2/extract_features.py"
SESSION=care-expanded2017-v2-smoke
test -f "$OUT/dataset_audit.json"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session already exists: $SESSION" >&2
  exit 2
fi
mkdir -p "$OUT/tmp" "$OUT/cache_home"
tmux new-session -d -s "$SESSION" \
  "cd '$PROJECT' && env PYTHONDONTWRITEBYTECODE=1 TMPDIR='$OUT/tmp' XDG_CACHE_HOME='$OUT/cache_home' EXPANDED_OUTPUT='$OUT' /home/boyuanjie/envs/tif/bin/python '$SCRIPT' --smoke-test --resume --workers 1 > '$OUT/smoke.console.log' 2>&1"
echo "Started $SESSION; monitor: tail -F $OUT/smoke.console.log"
