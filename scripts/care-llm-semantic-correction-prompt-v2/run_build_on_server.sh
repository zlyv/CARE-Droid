#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PYTHON_BIN="${CARE_ML_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
PACKAGE="scripts/care-llm-semantic-correction-prompt-v2"
"$PYTHON_BIN" "$PACKAGE/selftest.py"
"$PYTHON_BIN" "$PACKAGE/preflight.py"
"$PYTHON_BIN" "$PACKAGE/build_prompts_v2.py"
echo "PROMPT_V2_LOCAL_BUILD_PIPELINE_DONE_NO_LLM_CALL"
