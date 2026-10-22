#!/usr/bin/env bash
set -euo pipefail
SESSION=care_hprn_w0_manifest_auto_v1
ROOT=/home/boyuanjie/projects/tasa-droid
LOG="$ROOT/outputs/care_hprn_w0_manifest_auto_v1.console.log"
tmux new-session -d -s "$SESSION" "cd '$ROOT' && bash scripts/care-hprn-w0-manifest-auto-v1/resume_run.sh > '$LOG' 2>&1"
tmux has-session -t "$SESSION"
echo TMUX_OK
