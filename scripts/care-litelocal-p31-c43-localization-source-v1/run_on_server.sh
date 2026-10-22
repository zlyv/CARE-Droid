#!/usr/bin/env bash
set -euo pipefail

cd /home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python
SCRIPT=scripts/care-litelocal-p31-c43-localization-source-v1
ROOT=artifacts/care-litelocal-p31-c43-localization-source-v1
LOGS="$ROOT/logs"
mkdir -p "$LOGS" "$ROOT/arms" "$ROOT/results"

"$PY" "$SCRIPT/selftest.py"
"$PY" "$SCRIPT/preflight.py"

run_arm_job() {
  local arm="$1"
  local gpu="$2"
  local output="$ROOT/arms/$arm/source_predictions.csv"
  local log="$LOGS/${arm}.gpu${gpu}.log"
  if [[ -s "$output" ]]; then
    echo "[resume] $arm already complete"
    return 0
  fi
  echo "[run] $arm on CUDA_VISIBLE_DEVICES=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$SCRIPT/run_arm.py" \
    --arm "$arm" --device cuda --seeds 0,1,2 --jobs 4 \
    >"$log" 2>&1
  echo "[done] $arm on GPU$gpu"
}

run_arm_job L1_TOPK3_RAW 0 &
pid0=$!
run_arm_job L2_ARGMAX_ALIGNED 1 &
pid1=$!
failed=0
wait "$pid0" || failed=1
wait "$pid1" || failed=1
if [[ "$failed" -ne 0 ]]; then
  echo "BLOCKED_BY_CONTRACT: phase-1 arm failed; inspect $LOGS" >&2
  exit 1
fi

# The remaining interaction arm runs after both marginal arms.  This avoids
# concurrent writers and preserves one complete three-seed ensemble per arm.
run_arm_job L3_TOPK3_ALIGNED 0

"$PY" "$SCRIPT/summarize.py"
echo "P31_C43_LOCALIZATION_FULL_RUN_DONE"
