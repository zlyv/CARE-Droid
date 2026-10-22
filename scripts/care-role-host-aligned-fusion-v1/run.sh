#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/boyuanjie/projects/tasa-droid
HERE=$ROOT/scripts/care-role-host-aligned-fusion-v1
PY=/home/boyuanjie/envs/tif/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
echo RHAF_PIPELINE_START device=cuda years=2017-2021 dynamic=0 new_apks=0
$PY -m py_compile "$HERE"/*.py
$PY "$HERE/selftest.py"
$PY "$HERE/train.py" --preflight --resume
$PY "$HERE/train.py" --smoke-test --device cuda
$PY "$HERE/train.py" --train-source --resume --device cuda
$PY "$HERE/train.py" --train-rolling --resume --device cuda
$PY "$HERE/evaluate.py"
echo RHAF_PIPELINE_DONE
