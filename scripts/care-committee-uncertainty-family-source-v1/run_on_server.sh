#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY="/home/boyuanjie/envs/tif/bin/python"
if [[ ! -x "$PY" ]]; then PY="$(command -v python3 || true)"; fi
[[ -n "$PY" ]] || { echo "BLOCKED_BY_CONTRACT: no Python"; exit 2; }
"$PY" scripts/care-committee-uncertainty-family-source-v1/selftest.py
"$PY" scripts/care-committee-uncertainty-family-source-v1/preflight.py
"$PY" scripts/care-committee-uncertainty-family-source-v1/run_source_audit.py

