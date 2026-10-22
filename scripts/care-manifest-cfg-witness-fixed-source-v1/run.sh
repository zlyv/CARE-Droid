#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/boyuanjie/projects/tasa-droid
HERE=$ROOT/scripts/care-manifest-cfg-witness-fixed-source-v1
PY=/home/boyuanjie/envs/tif/bin/python
export PYTHONDONTWRITEBYTECODE=1
echo MANIFEST_CFG_WITNESS_FIXED_SOURCE_START
$PY -m py_compile "$HERE/evaluate.py"
$PY "$HERE/evaluate.py"
echo MANIFEST_CFG_WITNESS_FIXED_SOURCE_PIPELINE_DONE
