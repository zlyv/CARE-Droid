#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY=${PYTHON_BIN:-/home/boyuanjie/envs/tif/bin/python}
test -x "$PY" || { echo "BLOCKED_BY_CONTRACT: missing Python $PY"; exit 2; }
"$PY" scripts/care-viewdi-router-source-v1/selftest.py
"$PY" scripts/care-viewdi-router-source-v1/preflight.py
"$PY" scripts/care-viewdi-router-source-v1/run_viewdi_source.py

