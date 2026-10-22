#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python
"$PY" scripts/care-di25-hostego-r4-component-audit-v1/preflight.py
"$PY" scripts/care-di25-hostego-r4-component-audit-v1/run_component_audit.py
cat artifacts/care-di25-hostego-dmd-bounded-v1/r4_component_audit/R4_COMPONENT_AUDIT_REPORT.txt
