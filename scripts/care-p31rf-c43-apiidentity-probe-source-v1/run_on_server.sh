#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON_BIN="${PYTHON_BIN:-/home/boyuanjie/envs/tif/bin/python}"
export PYTHONHASHSEED=0
"$PYTHON_BIN" scripts/care-p31rf-c43-apiidentity-probe-source-v1/selftest.py
"$PYTHON_BIN" scripts/care-p31rf-c43-apiidentity-probe-source-v1/preflight.py
"$PYTHON_BIN" scripts/care-p31rf-c43-apiidentity-probe-source-v1/run_probe.py
"$PYTHON_BIN" scripts/care-p31rf-c43-apiidentity-probe-source-v1/summarize.py
echo P31RF_C43_APIIDENTITY_FULL_RUN_DONE

