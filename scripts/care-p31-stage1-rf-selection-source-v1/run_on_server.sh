#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python
ROOT=artifacts/care-p31-stage1-rf-selection-source-v1
SCRIPT=scripts/care-p31-stage1-rf-selection-source-v1
mkdir -p "$ROOT/arms" "$ROOT/logs" "$ROOT/results"
"$PY" "$SCRIPT/selftest.py"
"$PY" "$SCRIPT/build_references.py"
"$PY" "$SCRIPT/preflight.py"
run_arm() {
  local arm="$1" gpu="$2"
  if [[ -s "$ROOT/arms/$arm/source_predictions.csv" ]]; then echo "[resume] $arm complete"; return 0; fi
  echo "[run] $arm on GPU$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$SCRIPT/run_arm.py" --arm "$arm" --device cuda --seeds 0,1,2 >"$ROOT/logs/${arm}.gpu${gpu}.log" 2>&1
  echo "[done] $arm on GPU$gpu"
}
run_arm B_RF_LRROUTE 0 & p0=$!
run_arm C_RF_RFROUTE 1 & p1=$!
status=0; wait "$p0" || status=1; wait "$p1" || status=1
if [[ "$status" -ne 0 ]]; then echo "BLOCKED_BY_CONTRACT: RF arms failed; inspect $ROOT/logs"; exit 1; fi
run_arm D_MEAN_MEANROUTE 0
"$PY" "$SCRIPT/summarize.py"
echo "P31_RF_SELECTION_SOURCE_FULL_RUN_DONE"
