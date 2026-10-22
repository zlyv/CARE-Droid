#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=1
cd /home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python
$PY scripts/care-host-bound-cfg-source-future-v3/run_experiment.py preflight
exec > >(tee outputs/care_host_bound_cfg_source_future_v3/logs/console.log) 2>&1
echo HBCFG_RUN_AFTER_PREFLIGHT
$PY scripts/care-host-bound-cfg-source-future-v3/run_experiment.py features
$PY scripts/care-host-bound-cfg-source-future-v3/run_experiment.py train --device cuda
$PY scripts/care-host-bound-cfg-source-future-v3/run_experiment.py future --device cuda
$PY scripts/care-host-bound-cfg-source-future-v3/run_experiment.py evaluate
