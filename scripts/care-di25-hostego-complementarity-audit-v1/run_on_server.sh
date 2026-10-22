#!/usr/bin/env bash
set -euo pipefail

cd /home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python

"$PY" scripts/care-di25-hostego-complementarity-audit-v1/selftest.py
"$PY" scripts/care-di25-hostego-complementarity-audit-v1/preflight.py
"$PY" scripts/care-di25-hostego-complementarity-audit-v1/run_host_complementarity_audit.py
