#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid

SESSION="care-host-complement-neural-v1"
LOG="artifacts/care-di25-host-complement-neural-v1.console.log"
DRY="artifacts/care-di25-host-complement-neural-v1/preflight/dryrun_audit.json"
PYTHON_BIN="${CARE_ML_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"

"$PYTHON_BIN" -c 'import json,pathlib; p=pathlib.Path("artifacts/care-di25-host-complement-neural-v1/preflight/dryrun_audit.json"); assert p.is_file() and json.loads(p.read_text(encoding="utf-8"))["passed"] is True' || {
  echo "BLOCKED_BY_CONTRACT: successful dry-run is required before tmux full run" >&2
  exit 1
}

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach: tmux attach -t $SESSION"
  exit 0
fi

mkdir -p artifacts/care-di25-host-complement-neural-v1
tmux new-session -d -s "$SESSION" "cd /home/boyuanjie/projects/tasa-droid && bash scripts/care-di25-host-complement-neural-v1/run_on_server.sh 2>&1 | tee '$LOG'"
echo "Started: $SESSION"
echo "Monitor: tail -F $LOG"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"

