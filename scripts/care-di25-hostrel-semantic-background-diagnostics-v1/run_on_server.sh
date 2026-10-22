#!/usr/bin/env bash
set -euo pipefail
PROJECT="/home/boyuanjie/projects/tasa-droid"
PY="/home/boyuanjie/envs/tif/bin/python"
CODE="$PROJECT/scripts/care-di25-hostrel-semantic-background-diagnostics-v1"
ROOT="$PROJECT/artifacts/care-di25-hostrel-semantic-background-diagnostics-v1"
LOGS="$ROOT/logs"
cd "$PROJECT"
mkdir -p "$LOGS" "$ROOT/frozen" "$ROOT/within"
[[ -x "$PY" ]] || { echo "BLOCKED_BY_CONTRACT: missing Python $PY"; exit 2; }

"$PY" "$CODE/selftest.py"
"$PY" "$CODE/preflight.py"
"$PY" "$CODE/build_future_caches.py"

SOURCE_ENDPOINT="artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz"
FUTURE_ENDPOINT="artifacts/care-future-local-assets-v2/endpoint_centered_tensor_cache_2018_2021.jsonl.gz"
[[ -s "$FUTURE_ENDPOINT" ]] || FUTURE_ENDPOINT="/mnt/HDD1/boyuanjie/tasa-droid-storage/care-future-local-assets-v2/endpoint_centered_tensor_cache_2018_2021.jsonl.gz"
ARMS=(B1_HARD_ROLE R1_HOSTREL74_HARD_ROLE)

run_frozen() {
  local arm="$1"
  local gpu="$2"
  local out="$ROOT/frozen/$arm"
  mkdir -p "$out"
  if [[ -s "$out/future_predictions_label_free.csv" && -s "$out/protocol.json" ]]; then echo "[resume] frozen $arm complete"; return 0; fi
  echo "[run] frozen $arm on GPU$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$CODE/run_frozen_arm.py" --semantic-arm "$arm" --device cuda --seeds 900000,900001,900002 --source-cache "$SOURCE_ENDPOINT" --future-cache "$FUTURE_ENDPOINT" --source-regions "artifacts/care-source-local-assets-v2/regions_2017.npz" --future-regions "artifacts/care-future-local-assets-v2/regions_2018_2021.npz" --max-delta 2.25 --output "$out" 2>&1 | tee "$LOGS/frozen_${arm}.gpu${gpu}.log"
}

run_within() {
  local arm="$1"
  local gpu="$2"
  local out="$ROOT/within/$arm"
  mkdir -p "$out"
  if [[ -s "$out/within_year_predictions.csv" && -s "$out/protocol.json" ]]; then echo "[resume] within $arm complete"; return 0; fi
  echo "[run] within $arm on GPU$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$CODE/run_within_arm.py" --semantic-arm "$arm" --arm Q_QUERY_ONLY --device cuda --future-regions "$ROOT/caches/${arm}_regions_2018_2021.npz" --seeds 900000,900001,900002 --max-delta 2.25 --output "$out" 2>&1 | tee "$LOGS/within_${arm}.gpu${gpu}.log"
}

pids=(); labels=()
run_frozen B1_HARD_ROLE 0 & pids+=("$!"); labels+=("frozen/B1/GPU0")
run_within B1_HARD_ROLE 0 & pids+=("$!"); labels+=("within/B1/GPU0")
run_frozen R1_HOSTREL74_HARD_ROLE 1 & pids+=("$!"); labels+=("frozen/R1/GPU1")
run_within R1_HOSTREL74_HARD_ROLE 1 & pids+=("$!"); labels+=("within/R1/GPU1")
failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "[done] ${labels[$i]}"; else echo "[failed] ${labels[$i]}" >&2; failed=1; fi
done
(( failed == 0 )) || { echo "BLOCKED_BY_CONTRACT: diagnostic job failed; inspect $LOGS"; exit 3; }
"$PY" "$CODE/summarize.py"
echo "HOSTREL_SEMANTIC_DIAGNOSTICS_FULL_RUN_DONE"
