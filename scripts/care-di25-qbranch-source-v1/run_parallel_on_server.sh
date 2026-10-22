#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON_BIN="${CARE_ML_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
ROOT="artifacts/care-di25-qbranch-source-v1"
SCRIPT="scripts/care-di25-qbranch-source-v1"
mkdir -p "$ROOT/logs"

test -x "$PYTHON_BIN" || { echo "BLOCKED_BY_CONTRACT: missing Python $PYTHON_BIN"; exit 2; }
"$PYTHON_BIN" "$SCRIPT/selftest.py"
"$PYTHON_BIN" "$SCRIPT/preflight.py"

run_arm() {
  local arm="$1"
  local arm_root="$ROOT/arms/$arm"
  if "$PYTHON_BIN" "$SCRIPT/validate_complete.py" "$arm_root" >/dev/null 2>&1; then
    echo "[resume] $arm already complete"
    return 0
  fi
  echo "[run] $arm on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  "$PYTHON_BIN" "$SCRIPT/run_arm.py" --arm "$arm" --device cuda \
    2>&1 | tee "$ROOT/logs/${arm}.log"
}

# Two independent sequential queues prevent duplicate GPU work and keep memory bounded.
(
  export CUDA_VISIBLE_DEVICES=0
  run_arm ROLE_ONLY
  run_arm STRUCTURE_ONLY_Q
  run_arm STAGE1_CONTEXT_ONLY
) > >(tee "$ROOT/logs/gpu0.log") 2>&1 &
pid0=$!

(
  export CUDA_VISIBLE_DEVICES=1
  run_arm HOST_ONLY_Q
  run_arm HOST_ONLY_N0
) > >(tee "$ROOT/logs/gpu1.log") 2>&1 &
pid1=$!

set +e
wait "$pid0"; status0=$?
wait "$pid1"; status1=$?
set -e
if [[ "$status0" -ne 0 || "$status1" -ne 0 ]]; then
  echo "BLOCKED_BY_CONTRACT: branch arm failed gpu0=$status0 gpu1=$status1; inspect $ROOT/logs"
  exit 3
fi

"$PYTHON_BIN" "$SCRIPT/summarize.py"
echo "QBRANCH_SOURCE_EXPERIMENT_DONE"
