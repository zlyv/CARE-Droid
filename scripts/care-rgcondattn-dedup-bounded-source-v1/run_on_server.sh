#!/usr/bin/env bash
set -euo pipefail

cd /home/boyuanjie/projects/tasa-droid

PYTHON_BIN="${PYTHON_BIN:-/home/boyuanjie/envs/tif/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/care-rgcondattn-dedup-bounded-source-v1}"

export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"$PYTHON_BIN" \
  scripts/care-rgcondattn-dedup-bounded-source-v1/selftest_dedup.py

"$PYTHON_BIN" \
  scripts/care-rgcondattn-dedup-bounded-source-v1/run_care_rgcondattn_dedup_bounded_source_v1.py \
  --variants RGCondAttn \
  --seeds 0,1,2 \
  --device auto \
  --output "$OUTPUT_DIR"

cat "$OUTPUT_DIR/REPORT.txt"
