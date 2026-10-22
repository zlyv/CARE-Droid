#!/usr/bin/env bash
# Resume only this isolated experiment's feature extraction; never rerun audit.
set -euo pipefail
PROJECT=/home/boyuanjie/projects/tasa-droid
SCRIPT="$PROJECT/scripts/care-expanded2017-benign-aware-v2"
OUT="$PROJECT/outputs/care_expanded2017_benign_aware_v2"
PYTHON=/home/boyuanjie/envs/tif/bin/python
export EXPANDED_OUTPUT="$OUT"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$OUT/tmp"
export XDG_CACHE_HOME="$OUT/cache_home"
test -f "$OUT/dataset_audit.json"
test -f "$OUT/formal_p31_replay.json"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
"$PYTHON" "$SCRIPT/extract_features.py" --extract-all --resume --workers 1
"$PYTHON" "$SCRIPT/extract_features.py" --extract-stage2-train --resume --workers 1
