#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/boyuanjie/projects/tasa-droid
PKG="$PROJECT/scripts/care-di25-noego-within-year-replay-v1"
OUT="$PROJECT/artifacts/care-di25-noego-within-year-replay-v1"
PY="${CARE_ML_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"

cd "$PROJECT"
mkdir -p "$OUT/shared"

"$PY" "$PKG/selftest.py"
"$PY" "$PKG/preflight.py"
"$PY" "$PKG/run_replay.py"
"$PY" "$PKG/summarize.py"

echo NOEGO_WITHIN_YEAR_REPLAY_DONE
