#!/usr/bin/env bash
set -euo pipefail

cd /home/boyuanjie/projects/tasa-droid

PYTHON_BIN="${PYTHON_BIN:-/home/boyuanjie/envs/tif/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/care-di-directl2-future-diagnostic-v1}"

export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -f scripts/care-rgcondattn-direct-l2-source-v1/run_care_rgcondattn_direct_l2_source_v1.py ]]; then
  echo "Missing scripts/care-rgcondattn-direct-l2-source-v1/"
  echo "Upload and unzip care-rgcondattn-direct-l2-source-v1.zip first."
  exit 2
fi

if [[ ! -f artifacts/care-committee-decision-instability-router-future-v1/future_sample_scores.csv ]]; then
  echo "Missing DI future score artifact:"
  echo "  artifacts/care-committee-decision-instability-router-future-v1/future_sample_scores.csv"
  echo "Run care-committee-decision-instability-router-v1 first, then rerun this script."
  exit 2
fi

"$PYTHON_BIN" scripts/care-di-directl2-future-v1/selftest_di_directl2_future.py

"$PYTHON_BIN" \
  scripts/care-di-directl2-future-v1/run_di_directl2_future_v1.py \
  --years 2018,2019,2020,2021 \
  --variants RGConcat,RGCondAttn \
  --seeds 0,1,2 \
  --device auto \
  --output "$OUTPUT_DIR"

cat "$OUTPUT_DIR/REPORT.txt"
