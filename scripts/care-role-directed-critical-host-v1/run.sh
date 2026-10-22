#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=1
cd /home/boyuanjie/projects/tasa-droid
/home/boyuanjie/envs/tif/bin/python scripts/care-role-directed-critical-host-v1/run_experiment.py
