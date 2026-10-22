#!/usr/bin/env bash
# Isolated Phase A/B runner. Writes only to the fresh EXPANDED_OUTPUT directory.
set -euo pipefail

PROJECT=/home/boyuanjie/projects/tasa-droid
SCRIPT="$PROJECT/scripts/care-expanded2017-benign-aware-v2"
OUT="$PROJECT/outputs/care_expanded2017_benign_aware_v2"
PYTHON=/home/boyuanjie/envs/tif/bin/python

export EXPANDED_OUTPUT="$OUT"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$OUT/tmp"
export XDG_CACHE_HOME="$OUT/cache_home"

if [ -e "$OUT" ]; then
  echo "Refusing to start: output already exists. For resume use resume.sh; do not overwrite." >&2
  exit 2
fi
if [ ! -d "$SCRIPT" ]; then
  echo "Missing isolated script directory" >&2
  exit 2
fi

"$PYTHON" "$SCRIPT/audit_new_apks.py"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
"$PYTHON" "$SCRIPT/extract_features.py" --smoke-test --workers 1
"$PYTHON" "$SCRIPT/extract_features.py" --extract-all --resume --workers 1
"$PYTHON" "$SCRIPT/extract_features.py" --extract-stage2-train --resume --workers 1

echo "EXPANDED_PHASE_AB_COMPLETE: inspect coverage and compatibility before training" >&2
