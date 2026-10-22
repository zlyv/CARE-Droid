#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY="${CARE_ML_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
S=scripts/care-di25-host-relative-source-v1
ROOT=artifacts/care-di25-host-relative-source-v1
mkdir -p "$ROOT/logs" "$ROOT/arms" "$ROOT/results"
"$PY" "$S/selftest.py"
"$PY" "$S/preflight.py"
"$PY" "$S/build_caches.py"
run_arm() {
  local arm="$1"
  local gpu="$2"
  local log="$ROOT/logs/${arm}.gpu${gpu}.log"
  if [[ -s "$ROOT/arms/$arm/source_predictions.csv" ]]; then echo "[resume] $arm already complete"; return 0; fi
  echo "[run] $arm on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$S/run_arm.py" --arm "$arm" --seeds 0,1,2 --device cuda 2>&1 | tee "$log"
}
run_arm HM_R 0 & p0=$!
run_arm HC_R 1 & p1=$!
status=0
wait "$p0" || status=1
wait "$p1" || status=1
if [[ "$status" -ne 0 ]]; then
  echo "BLOCKED_BY_CONTRACT: HM_R or HC_R failed; inspect $ROOT/logs" >&2
  exit 1
fi
run_arm HMC_R 1
"$PY" "$S/summarize.py"
