#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PYTHON_BIN="${CARE_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
"$PYTHON_BIN" scripts/care-llm-semantic-correction-expert-pair-v1/run_expert_pilot.py --execute
"$PYTHON_BIN" scripts/care-llm-semantic-correction-expert-pair-v1/summarize_pair.py

