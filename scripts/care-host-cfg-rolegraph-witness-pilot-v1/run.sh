#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
ROOT=/home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python
HERE=$ROOT/scripts/care-host-cfg-rolegraph-witness-pilot-v1
echo "CARE_CFG_ROLEGRAPH_WITNESS_PILOT_START source=2017 routed=200 future=0 workers=${CFGW_WORKERS:-4}"
$PY -m py_compile "$HERE"/*.py
echo "CFGW_PHASE_DONE name=syntax"
$PY "$HERE/selftest.py"
echo "CFGW_PHASE_DONE name=selftest"
$PY "$HERE/pipeline.py" preflight
echo "CFGW_PHASE_DONE name=preflight"
$PY "$HERE/pipeline.py" scan --workers "${CFGW_WORKERS:-4}"
echo "CFGW_PHASE_DONE name=scan"
$PY "$HERE/pipeline.py" summarize
echo "CARE_CFG_ROLEGRAPH_WITNESS_PILOT_DONE"

