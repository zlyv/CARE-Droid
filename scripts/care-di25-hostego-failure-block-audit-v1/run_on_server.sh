#!/usr/bin/env bash
set -euo pipefail

cd /home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python

"$PY" scripts/care-di25-hostego-failure-block-audit-v1/selftest.py
"$PY" scripts/care-di25-hostego-failure-block-audit-v1/preflight.py
"$PY" scripts/care-di25-hostego-failure-block-audit-v1/run_failure_block_audit.py

cat artifacts/care-di25-hostego-dmd-bounded-v1/failure_block_audit/FAILURE_BLOCK_AUDIT_REPORT.txt
