#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python
ROOT=artifacts/care-litelocal-p31-c43-source-v1

"$PY" scripts/care-litelocal-p31-c43-source-v1/selftest.py
"$PY" scripts/care-litelocal-p31-c43-source-v1/build_p31_reference.py
"$PY" scripts/care-litelocal-p31-c43-source-v1/preflight.py
mkdir -p "$ROOT/arms/P31_C43_RQ" "$ROOT/results" "$ROOT/audits"

if [[ -s "$ROOT/arms/P31_C43_RQ/source_predictions.csv" ]]; then
  echo "[resume] P31_C43_RQ already complete"
else
  "$PY" scripts/care-litelocal-p31-c43-source-v1/run_p31_c43.py \
    --device cuda --seeds 0,1,2 --jobs 4
fi

"$PY" scripts/care-litelocal-p31-c43-source-v1/summarize.py
echo "P31_C43_SOURCE_FULL_RUN_DONE"
