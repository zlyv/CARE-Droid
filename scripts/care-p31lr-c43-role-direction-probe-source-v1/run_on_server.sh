#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON_BIN="${CARE_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
"$PYTHON_BIN" scripts/care-p31lr-c43-role-direction-probe-source-v1/selftest.py
"$PYTHON_BIN" scripts/care-p31lr-c43-role-direction-probe-source-v1/preflight.py
"$PYTHON_BIN" scripts/care-p31lr-c43-role-direction-probe-source-v1/run_probe.py
echo P31LR_C43_ROLE_DIRECTION_PROBE_FULL_RUN_DONE
