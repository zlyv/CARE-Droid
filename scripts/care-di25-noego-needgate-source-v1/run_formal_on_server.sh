#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/boyuanjie/projects/tasa-droid"
PYTHON_BIN="${CARE_ML_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
DEVICE="${CARE_DEVICE:-cuda:0}"
PACKAGE="$PROJECT_ROOT/scripts/care-di25-noego-needgate-source-v1"

cd "$PROJECT_ROOT"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "BLOCKED_BY_CONTRACT: missing executable Python: $PYTHON_BIN" >&2
  exit 2
fi

"$PYTHON_BIN" "$PACKAGE/selftest.py"
"$PYTHON_BIN" "$PACKAGE/preflight.py"

"$PYTHON_BIN" "$PACKAGE/run_needgate_source.py" --device "$DEVICE"
"$PYTHON_BIN" "$PACKAGE/summarize.py"

echo "SOURCE_NEEDGATE_FORMAL_PIPELINE_DONE"
