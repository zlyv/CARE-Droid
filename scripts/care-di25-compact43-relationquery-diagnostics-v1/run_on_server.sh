#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY="/home/boyuanjie/envs/tif/bin/python"
CODE="scripts/care-di25-compact43-relationquery-diagnostics-v1"
ROOT="artifacts/care-di25-compact43-relationquery-diagnostics-v1"
LOGS="$ROOT/logs"
mkdir -p "$LOGS" "$ROOT/frozen/C43_RELATION_QUERY" "$ROOT/within/C43_RELATION_QUERY"
[[ -x "$PY" ]] || { echo "BLOCKED_BY_CONTRACT: missing $PY"; exit 2; }
"$PY" "$CODE/selftest.py"
"$PY" "$CODE/preflight.py"
"$PY" "$CODE/build_future_cache.py"

run_frozen() {
  local out="$ROOT/frozen/C43_RELATION_QUERY"
  if [[ -s "$out/future_predictions_label_free.csv" ]]; then echo "[resume] frozen C43_RELATION_QUERY complete"; return 0; fi
  CUDA_VISIBLE_DEVICES=0 "$PY" "$CODE/run_frozen.py" \
    --device cuda --seeds 900000,900001,900002 --max-delta 2.25 \
    --output "$out" 2>&1 | tee "$LOGS/frozen_C43_RELATION_QUERY.gpu0.log"
}

run_within() {
  local out="$ROOT/within/C43_RELATION_QUERY"
  if [[ -s "$out/within_year_predictions.csv" ]]; then echo "[resume] within C43_RELATION_QUERY complete"; return 0; fi
  CUDA_VISIBLE_DEVICES=1 "$PY" "$CODE/run_within.py" \
    --device cuda --seeds 900000,900001,900002 --max-delta 2.25 \
    --output "$out" 2>&1 | tee "$LOGS/within_C43_RELATION_QUERY.gpu1.log"
}

run_frozen & p0=$!
run_within & p1=$!
status=0
wait "$p0" || status=1
wait "$p1" || status=1
[[ "$status" -eq 0 ]] || { echo "BLOCKED_BY_CONTRACT: diagnostic job failed; inspect $LOGS"; exit 3; }
"$PY" "$CODE/summarize.py"
echo "COMPACT43_RELATIONQUERY_DIAGNOSTICS_FULL_RUN_DONE"
