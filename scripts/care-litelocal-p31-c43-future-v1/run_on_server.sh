#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python
DIR=scripts/care-litelocal-p31-c43-future-v1
ROOT=artifacts/care-litelocal-p31-c43-future-v1

"$PY" "$DIR/selftest.py"

P31_ARGS=()
if [[ -n "${FUTURE_P31_CACHE:-}" ]]; then
  P31_ARGS+=(--future-p31-cache "$FUTURE_P31_CACHE")
fi
"$PY" "$DIR/build_p31_future_reference.py" "${P31_ARGS[@]}"
"$PY" "$DIR/prepare_eval_features.py"
"$PY" "$DIR/preflight.py"
mkdir -p "$ROOT/arm/P31_C43_RQ" "$ROOT/results"

if [[ -s "$ROOT/arm/P31_C43_RQ/future_predictions_diagnostic.csv" ]]; then
  echo "[resume] future Stage-II prediction already exists"
else
  "$PY" "$DIR/run_p31_c43_future.py" --device cuda --seeds 0,1,2
fi
"$PY" "$DIR/summarize_future.py"
echo "P31_C43_FUTURE_FULL_RUN_DONE"
