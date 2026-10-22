#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python
ROOT=artifacts/care-litelocal-p31lr-c43-prov6-source-v1
SCRIPT=scripts/care-litelocal-p31lr-c43-prov6-source-v1

"$PY" "$SCRIPT/selftest.py"
"$PY" "$SCRIPT/build_lr_reference.py"
"$PY" "$SCRIPT/preflight.py"
"$PY" "$SCRIPT/build_prov6_caches.py"
mkdir -p "$ROOT/arms" "$ROOT/logs" "$ROOT/results"

run_arm() {
  local arm="$1" gpu="$2"
  if [[ -s "$ROOT/arms/$arm/source_predictions.csv" ]]; then
    echo "[resume] $arm complete"
    return 0
  fi
  echo "[run] $arm on GPU$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$SCRIPT/run_arm.py" --arm "$arm" --device cuda --seeds 0,1,2 \
    >"$ROOT/logs/${arm}.gpu${gpu}.log" 2>&1
  echo "[done] $arm on GPU$gpu"
}

run_arm P0_ZERO6 0 & p0=$!
run_arm P1_PROV6 1 & p1=$!
status=0
wait "$p0" || status=1
wait "$p1" || status=1
if [[ "$status" -ne 0 ]]; then
  echo "BLOCKED_BY_CONTRACT: P0/P1 failed; inspect $ROOT/logs"
  exit 1
fi
run_arm P2_SHUFFLED_PROV6 0
"$PY" "$SCRIPT/summarize.py"
echo "P31LR_PROV6_SOURCE_FULL_RUN_DONE"
