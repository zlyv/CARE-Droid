#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python
ROOT=artifacts/care-litelocal-p31-c43-direct-source-v1

"$PY" scripts/care-litelocal-p31-c43-direct-source-v1/selftest.py
"$PY" scripts/care-litelocal-p31-c43-direct-source-v1/preflight.py
mkdir -p "$ROOT/arms/P31_C43_DIRECT" "$ROOT/results"

if [[ -s "$ROOT/arms/P31_C43_DIRECT/source_predictions.csv" ]]; then
  echo "[resume] P31_C43_DIRECT already complete"
else
  "$PY" scripts/care-litelocal-p31-c43-direct-source-v1/run_direct.py \
    --device cuda --seeds 0,1,2 --jobs 4
fi

"$PY" scripts/care-litelocal-p31-c43-direct-source-v1/summarize.py
echo "P31_C43_DIRECT_SOURCE_FULL_RUN_DONE"
