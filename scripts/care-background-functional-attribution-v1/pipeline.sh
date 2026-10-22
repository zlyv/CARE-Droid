#!/usr/bin/env bash
# BFA-V1 detached pipeline; only new outputs are written, no APKs opened or executed.
set -euo pipefail
ROOT=/home/boyuanjie/projects/tasa-droid
OUT="$ROOT/outputs/care_background_functional_attribution_v1"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TMPDIR="$OUT/tmp"
cd "$ROOT"
echo BFA_PIPELINE_START "$(date -u +%FT%TZ)"
nice -n 10 bash scripts/care-background-functional-attribution-v1/run.sh --attribution --resume --device cuda
nice -n 10 bash scripts/care-background-functional-attribution-v1/run.sh --full-attribution --resume --device cuda
nice -n 10 bash scripts/care-background-functional-attribution-v1/run.sh --train-source --resume --device cuda
nice -n 10 bash scripts/care-background-functional-attribution-v1/run.sh --train-rolling --resume --device cuda
nice -n 10 bash scripts/care-background-functional-attribution-v1/run.sh --evaluate --resume --device cpu
echo BFA_PIPELINE_DONE "$(date -u +%FT%TZ)"
