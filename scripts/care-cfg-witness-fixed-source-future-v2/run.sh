#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/boyuanjie/projects/tasa-droid
HERE=$ROOT/scripts/care-cfg-witness-fixed-source-future-v2
PY=/home/boyuanjie/envs/tif/bin/python
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
echo CFGW_FIXED_SOURCE_FUTURE_START device=${CFGW_FS_DEVICE:-cuda}
$PY -m py_compile "$HERE/run_fixed_source_future.py"
$PY "$HERE/run_fixed_source_future.py"
echo CFGW_FIXED_SOURCE_FUTURE_PIPELINE_DONE
