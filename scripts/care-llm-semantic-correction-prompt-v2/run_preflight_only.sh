#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PYTHON_BIN="${CARE_ML_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
PACKAGE="scripts/care-llm-semantic-correction-prompt-v2"
"$PYTHON_BIN" "$PACKAGE/selftest.py"
"$PYTHON_BIN" "$PACKAGE/preflight.py"
echo "PROMPT_V2_PREFLIGHT_ONLY_DONE"
