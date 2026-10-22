#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY="/home/boyuanjie/envs/tif/bin/python"
CODE="scripts/care-di25-hostrel-semantic-background-source-v1"
ROOT="artifacts/care-di25-hostrel-semantic-background-source-v1"
ARM="R1_HOSTREL74_HARD_ROLE"
GPU="${CARE_GPU:-0}"
mkdir -p "$ROOT/logs" "$ROOT/arms"
[[ -x "$PY" ]] || { echo "BLOCKED_BY_CONTRACT: missing $PY"; exit 2; }
"$PY" "$CODE/selftest.py"
"$PY" "$CODE/preflight.py"
"$PY" "$CODE/build_cache.py"
if [[ -s "$ROOT/arms/$ARM/source_predictions.csv" ]]; then
  echo "[resume] $ARM complete"
else
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$CODE/run_arm.py" --arm "$ARM" --seeds 0,1,2 --device cuda 2>&1 | tee "$ROOT/logs/${ARM}.gpu${GPU}.log"
fi
"$PY" "$CODE/summarize.py"
echo "HOSTREL_SEMANTIC_BACKGROUND_SOURCE_FULL_RUN_DONE"
