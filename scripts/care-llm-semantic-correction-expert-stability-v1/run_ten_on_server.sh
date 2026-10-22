#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PYTHON_BIN="${CARE_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
"$PYTHON_BIN" scripts/care-llm-semantic-correction-expert-stability-v1/run_stability.py --execute
"$PYTHON_BIN" scripts/care-llm-semantic-correction-expert-stability-v1/summarize_stability.py

