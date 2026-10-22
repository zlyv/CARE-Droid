#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python

"$PY" scripts/care-litelocal-p31-c43-failure-audit-v1/selftest.py
"$PY" scripts/care-litelocal-p31-c43-failure-audit-v1/preflight.py
"$PY" scripts/care-litelocal-p31-c43-failure-audit-v1/run_audit.py
