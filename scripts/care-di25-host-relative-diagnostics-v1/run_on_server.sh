#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/boyuanjie/projects/tasa-droid"
PY="/home/boyuanjie/envs/tif/bin/python"
CODE="$PROJECT/scripts/care-di25-host-relative-diagnostics-v1"
ROOT="$PROJECT/artifacts/care-di25-host-relative-diagnostics-v1"
LOGS="$ROOT/logs"
cd "$PROJECT"
mkdir -p "$LOGS" "$ROOT/frozen" "$ROOT/within"

if [[ ! -x "$PY" ]]; then
  echo "BLOCKED_BY_CONTRACT: missing Python $PY" >&2
  exit 2
fi

"$PY" "$CODE/selftest.py"
"$PY" "$CODE/preflight.py"
"$PY" "$CODE/build_future_caches.py"

SOURCE_ENDPOINT="artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz"
FUTURE_ENDPOINT="artifacts/care-future-local-assets-v2/endpoint_centered_tensor_cache_2018_2021.jsonl.gz"
if [[ ! -s "$FUTURE_ENDPOINT" ]]; then
  FUTURE_ENDPOINT="/mnt/HDD1/boyuanjie/tasa-droid-storage/care-future-local-assets-v2/endpoint_centered_tensor_cache_2018_2021.jsonl.gz"
fi

run_frozen() {
  local arm="$1"
  local gpu="$2"
  local out="$ROOT/frozen/$arm"
  mkdir -p "$out"
  if [[ -s "$out/future_predictions_label_free.csv" && -s "$out/protocol.json" ]]; then
    echo "[resume] frozen $arm already complete"
    return 0
  fi
  echo "[run] frozen $arm on GPU$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$CODE/run_frozen_arm.py" \
    --arm "$arm" --device cuda --seeds 900000,900001,900002 \
    --source-cache "$SOURCE_ENDPOINT" --future-cache "$FUTURE_ENDPOINT" \
    --source-regions "artifacts/care-di25-host-relative-source-v1/caches/${arm}_regions_2017.npz" \
    --future-regions "$ROOT/caches/${arm}_regions_2018_2021.npz" \
    --max-delta 2.25 --output "$out" \
    2>&1 | tee "$LOGS/frozen_${arm}.gpu${gpu}.log"
}

run_within() {
  local arm="$1"
  local gpu="$2"
  local out="$ROOT/within/$arm"
  mkdir -p "$out"
  if [[ -s "$out/within_year_predictions.csv" && -s "$out/protocol.json" ]]; then
    echo "[resume] within $arm already complete"
    return 0
  fi
  echo "[run] within $arm on GPU$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$CODE/run_within_arm.py" \
    --host-arm "$arm" --arm Q_QUERY_ONLY --device cuda \
    --future-regions "$ROOT/caches/${arm}_regions_2018_2021.npz" \
    --seeds 900000,900001,900002 --max-delta 2.25 --output "$out" \
    2>&1 | tee "$LOGS/within_${arm}.gpu${gpu}.log"
}

run_phase() {
  local phase="$1"
  shift
  local pids=()
  local labels=()
  while (( "$#" )); do
    local kind="$1"
    local arm="$2"
    local gpu="$3"
    shift 3
    if [[ "$kind" == "frozen" ]]; then
      run_frozen "$arm" "$gpu" &
    else
      run_within "$arm" "$gpu" &
    fi
    pids+=("$!")
    labels+=("$kind/$arm/GPU$gpu")
  done
  local failed=0
  local i
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "[failed] $phase ${labels[$i]}" >&2
      failed=1
    else
      echo "[done] $phase ${labels[$i]}"
    fi
  done
  if (( failed )); then
    echo "BLOCKED_BY_CONTRACT: $phase has failed jobs; inspect $LOGS" >&2
    exit 3
  fi
}

# Both RTX 3080s are used concurrently.  Each model is small, so phase 1 runs
# one frozen and one within-year job per GPU; phase 2 runs the two HMC jobs.
run_phase phase1 \
  frozen HM_R 0 \
  within HM_R 0 \
  frozen HC_R 1 \
  within HC_R 1

run_phase phase2 \
  frozen HMC_R 0 \
  within HMC_R 1

"$PY" "$CODE/summarize_frozen.py"
"$PY" "$CODE/summarize_within.py"
"$PY" "$CODE/summarize_combined.py"
echo "HOST_RELATIVE_DIAGNOSTICS_FULL_RUN_DONE"
