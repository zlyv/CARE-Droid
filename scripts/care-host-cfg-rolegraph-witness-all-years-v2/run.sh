#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
ROOT=/home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python
HERE=$ROOT/scripts/care-host-cfg-rolegraph-witness-all-years-v2
echo "CARE_CFG_ROLEGRAPH_WITNESS_ALL_YEARS_START apks=2788 routed=697 years=2017-2021 workers=${CFGW_WORKERS:-4}"
$PY -m py_compile "$HERE"/*.py
echo "CFGW_PHASE_DONE name=syntax"
$PY "$HERE/selftest.py"
echo "CFGW_PHASE_DONE name=selftest"
$PY "$HERE/pipeline.py" preflight
echo "CFGW_PHASE_DONE name=preflight"
$PY "$HERE/pipeline.py" scan --workers "${CFGW_WORKERS:-4}"
echo "CFGW_PHASE_DONE name=scan"
$PY "$HERE/pipeline.py" summarize
echo "CARE_CFG_ROLEGRAPH_WITNESS_ALL_YEARS_DONE"

