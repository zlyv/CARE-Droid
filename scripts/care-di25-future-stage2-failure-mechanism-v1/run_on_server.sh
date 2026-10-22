#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python
[[ -x "$PY" ]] || PY="$(command -v python3 || true)"
[[ -n "$PY" ]] || { echo "BLOCKED_BY_CONTRACT: no Python"; exit 2; }
"$PY" scripts/care-di25-future-stage2-failure-mechanism-v1/selftest.py
"$PY" scripts/care-di25-future-stage2-failure-mechanism-v1/preflight.py
"$PY" scripts/care-di25-future-stage2-failure-mechanism-v1/run_audit.py

