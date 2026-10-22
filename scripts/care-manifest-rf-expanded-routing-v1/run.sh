#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/boyuanjie/projects/tasa-droid
HERE=$ROOT/scripts/care-manifest-rf-expanded-routing-v1
PY=/home/boyuanjie/envs/tif/bin/python
export PYTHONDONTWRITEBYTECODE=1
echo MANIFEST_RF_EXPANDED_ROUTING_AUDIT_START
$PY -m py_compile "$HERE/evaluate.py"
$PY "$HERE/evaluate.py"
echo MANIFEST_RF_EXPANDED_ROUTING_AUDIT_PIPELINE_DONE
