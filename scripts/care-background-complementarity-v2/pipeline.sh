#!/usr/bin/env bash
# BFA-V2 resumable tmux worker; no old file modification, no APK parsing.
set -euo pipefail
ROOT=/home/boyuanjie/projects/tasa-droid
cd "$ROOT"
SCRIPT=scripts/care-background-complementarity-v2/run.sh
echo BFA_V2_PIPELINE_START
bash "$SCRIPT" --audit-only --resume
bash "$SCRIPT" --analyze-existing --resume
bash "$SCRIPT" --smoke-test --device cuda --resume
bash "$SCRIPT" --train-source --device cuda --resume
bash "$SCRIPT" --train-rolling --device cuda --resume
bash "$SCRIPT" --evaluate --resume
bash "$SCRIPT" --analyze-complementarity --resume
bash "$SCRIPT" --distribution-shift --device cuda --resume
bash "$SCRIPT" --collect --resume
echo BFA_V2_PIPELINE_DONE
