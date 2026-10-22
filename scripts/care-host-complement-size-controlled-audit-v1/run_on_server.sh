#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python
S=scripts/care-host-complement-size-controlled-audit-v1
mkdir -p artifacts/care-host-complement-size-controlled-audit-v1/{preflight,predictions,reports,audit}
"$PY" "$S/selftest.py"
"$PY" "$S/preflight.py"
"$PY" "$S/run_size_controlled_audit.py"
"$PY" "$S/summarize.py"

