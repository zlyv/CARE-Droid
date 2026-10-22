#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=1
cd /home/boyuanjie/projects/tasa-droid
/home/boyuanjie/envs/tif/bin/python scripts/care-hprn-w0-manifest-auto-v1/train.py --preflight --resume
/home/boyuanjie/envs/tif/bin/python scripts/care-hprn-w0-manifest-auto-v1/train.py --train-source --resume --device cuda
/home/boyuanjie/envs/tif/bin/python scripts/care-hprn-w0-manifest-auto-v1/train.py --train-full2017 --resume --device cuda
/home/boyuanjie/envs/tif/bin/python scripts/care-hprn-w0-manifest-auto-v1/evaluate.py --device cuda
echo HPRN_PIPELINE_DONE
