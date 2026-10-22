#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python
ROOT=artifacts/care-di25-graphglobal-path-source-v1

"$PY" scripts/care-di25-graphglobal-path-source-v1/selftest.py
"$PY" scripts/care-di25-graphglobal-path-source-v1/preflight.py
mkdir -p "$ROOT/arms" "$ROOT/results"

ARMS=(N0_NO_GRAPHGLOBAL D_DELTA_ONLY Q_QUERY_ONLY QD_QUERY_AND_DELTA)
for ARM in "${ARMS[@]}"; do
  PRED="$ROOT/arms/$ARM/source_predictions.csv"
  if [[ -s "$PRED" ]]; then
    echo "[resume] $ARM already complete"
  else
    echo "[run] $ARM"
    "$PY" scripts/care-di25-graphglobal-path-source-v1/run_arm.py \
      --arm "$ARM" --seeds 0,1,2 --device auto
  fi
done

"$PY" scripts/care-di25-graphglobal-path-source-v1/summarize.py

