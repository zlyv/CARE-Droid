#!/usr/bin/env bash
set -euo pipefail

cd /home/boyuanjie/projects/tasa-droid
PYTHON_BIN="${PYTHON_BIN:-/home/boyuanjie/envs/tif/bin/python}"
export PYTHONNOUSERSITE=1

"$PYTHON_BIN" \
  scripts/care-asymmetric-a25-router-source-v1/selftest_asymmetric_a25.py

"$PYTHON_BIN" \
  scripts/care-asymmetric-a25-router-source-v1/run_asymmetric_a25_router_source_v1.py \
  --output artifacts/care-asymmetric-a25-router-source-v1

cat artifacts/care-asymmetric-a25-router-source-v1/REPORT.txt
