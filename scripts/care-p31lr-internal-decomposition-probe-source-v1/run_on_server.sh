#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python
SCRIPT=scripts/care-p31lr-internal-decomposition-probe-source-v1
"$PY" "$SCRIPT/selftest.py"
"$PY" "$SCRIPT/preflight.py"
"$PY" "$SCRIPT/run_probe.py"
echo P31LR_INTERNAL_DECOMPOSITION_PROBE_FULL_RUN_DONE
