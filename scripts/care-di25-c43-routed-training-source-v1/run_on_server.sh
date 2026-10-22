#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY="/home/boyuanjie/envs/tif/bin/python"
CODE="scripts/care-di25-c43-routed-training-source-v1"
ROOT="artifacts/care-di25-c43-routed-training-source-v1"
mkdir -p "$ROOT/logs" "$ROOT/arms"
[[ -x "$PY" ]] || { echo "BLOCKED_BY_CONTRACT: missing $PY"; exit 2; }
"$PY" "$CODE/selftest.py"
"$PY" "$CODE/preflight.py"
run_arm() {
  local arm="$1"
  local gpu="$2"
  if [[ -s "$ROOT/arms/$arm/source_predictions.csv" && -s "$ROOT/arms/$arm/per_seed_predictions.csv" ]]; then echo "[resume] $arm complete"; return 0; fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$CODE/run_arm.py" --arm "$arm" --seeds 0,1,2 --device cuda 2>&1 | tee "$ROOT/logs/${arm}.gpu${gpu}.log"
}
run_arm T1_ROUTED_LOSS 0 & p0=$!
run_arm T2_FULLY_ROUTED 1 & p1=$!
status=0;wait "$p0"||status=1;wait "$p1"||status=1
[[ "$status" -eq 0 ]] || { echo "BLOCKED_BY_CONTRACT: routed-training arm failed; inspect $ROOT/logs"; exit 3; }
"$PY" "$CODE/summarize.py"
echo "C43_ROUTED_TRAINING_SOURCE_FULL_RUN_DONE"

