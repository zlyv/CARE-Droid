#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
ROOT=/home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python
HERE="$ROOT/scripts/care-internal-self-reference-audit-v2"
OUT="$ROOT/outputs/care_internal_self_reference_audit_v2"
echo "CARE_INTERNAL_SELF_REFERENCE_AUDIT_START output=$OUT device=${ISR_DEVICE:-cpu}"
"$PY" -m py_compile "$HERE"/*.py
echo "ISR_PHASE_DONE name=syntax"
"$PY" "$HERE/selftest.py"
echo "ISR_PHASE_DONE name=selftest"
"$PY" "$HERE/run_audit.py" --device "${ISR_DEVICE:-cpu}"

