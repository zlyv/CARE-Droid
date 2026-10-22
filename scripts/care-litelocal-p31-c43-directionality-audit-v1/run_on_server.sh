#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python
SCRIPT=scripts/care-litelocal-p31-c43-directionality-audit-v1

"$PY" "$SCRIPT/selftest.py"
"$PY" "$SCRIPT/preflight.py"
"$PY" "$SCRIPT/run_audit.py"
echo "P31_C43_DIRECTIONALITY_AUDIT_FULL_RUN_DONE"
