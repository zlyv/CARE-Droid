#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python
ROOT=artifacts/care-di25-host-input-ablation-v1

"$PY" scripts/care-di25-host-input-ablation-v1/preflight.py
"$PY" scripts/care-di25-host-input-ablation-v1/selftest.py
mkdir -p "$ROOT/arms" "$ROOT/results"

ARMS=(FULL DROP_GLOBAL24 DROP_STAGE1_LOGIT DROP_ROLE_GRAPH DROP_CRITICAL_HOST DROP_ATTENDED_HOST DROP_SCALAR_SUMMARY STAGE1_CONTEXT_ONLY STRUCTURE_ONLY)
for ARM in "${ARMS[@]}"; do
  PRED="$ROOT/arms/$ARM/source_predictions.csv"
  if [[ -s "$PRED" ]]; then
    echo "[resume] $ARM already complete"
  else
    echo "[run] $ARM"
    "$PY" scripts/care-di25-host-input-ablation-v1/run_arm.py --arm "$ARM" --seeds 0,1,2 --device auto
  fi
done

"$PY" scripts/care-di25-host-input-ablation-v1/summarize.py

