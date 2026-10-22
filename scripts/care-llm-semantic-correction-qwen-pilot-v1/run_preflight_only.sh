#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PYTHON_BIN="${CARE_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
"$PYTHON_BIN" scripts/care-llm-semantic-correction-qwen-pilot-v1/selftest.py
"$PYTHON_BIN" scripts/care-llm-semantic-correction-qwen-pilot-v1/preflight.py

