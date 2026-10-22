#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=1
cd /home/boyuanjie/projects/tasa-droid
/home/boyuanjie/envs/tif/bin/python scripts/care-hprn-w0-manifest-auto-v1/train.py --preflight
/home/boyuanjie/envs/tif/bin/python scripts/care-hprn-w0-manifest-auto-v1/train.py --smoke-test --device cuda
/home/boyuanjie/envs/tif/bin/python scripts/care-hprn-w0-manifest-auto-v1/train.py --train-source --device cuda
/home/boyuanjie/envs/tif/bin/python scripts/care-hprn-w0-manifest-auto-v1/train.py --train-full2017 --device cuda
/home/boyuanjie/envs/tif/bin/python scripts/care-hprn-w0-manifest-auto-v1/evaluate.py --device cuda
echo HPRN_PIPELINE_DONE
