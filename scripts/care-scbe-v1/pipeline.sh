#!/usr/bin/env bash
# Detached SCBE experiment pipeline; logs and temporary files only in new output.
set -euo pipefail
ROOT=/home/boyuanjie/projects/tasa-droid
OUT="$ROOT/outputs/care_scbe_v1"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1
export TMPDIR="$OUT/tmp"
mkdir -p "$TMPDIR"
cd "$ROOT"
echo SCBE_PIPELINE_START "$(date -u +%FT%TZ)"
nice -n 10 bash scripts/care-scbe-v1/run.sh --train-source --resume --device cuda --workers 1
echo SCBE_SOURCE_DONE "$(date -u +%FT%TZ)"
nice -n 10 bash scripts/care-scbe-v1/run.sh --train-rolling --resume --device cuda --workers 1
echo SCBE_ROLLING_DONE "$(date -u +%FT%TZ)"
nice -n 10 bash scripts/care-scbe-v1/run.sh --evaluate --resume --device cpu --workers 1
echo SCBE_PIPELINE_DONE "$(date -u +%FT%TZ)"
