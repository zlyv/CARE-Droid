#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python

# Deliberately calls only the post-hoc reader. It never calls the M0--M3 runner.
"$PY" scripts/care-di25-hostego-v1/audit_m01_only.py

echo "Audit-only outputs: artifacts/care-di25-hostego-v1/audits/AUDIT_M01_REPORT.txt"
