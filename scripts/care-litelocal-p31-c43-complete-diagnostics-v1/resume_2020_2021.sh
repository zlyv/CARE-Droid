#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid

PY=/home/boyuanjie/envs/tif/bin/python
SCRIPT=scripts/care-litelocal-p31-c43-complete-diagnostics-v1
ROOT=artifacts/care-litelocal-p31-c43-complete-diagnostics-v1
LOGS="$ROOT/logs"
mkdir -p "$LOGS" "$ROOT/within_year/2020_2021" "$ROOT/results"

"$PY" "$SCRIPT/selftest.py"
"$PY" "$SCRIPT/preflight.py"

CUDA_VISIBLE_DEVICES=0 "$PY" "$SCRIPT/run_within_year.py" \
  --representation anomaly --head bounded --device cuda \
  --seeds 900000,900001,900002 --max-delta 2.25 \
  --years 2020,2021 --output "$ROOT/within_year/2020_2021" \
  2>&1 | tee "$LOGS/within_2020_2021.gpu0.log"

"$PY" "$SCRIPT/summarize.py"
echo "P31_C43_COMPLETE_DIAGNOSTICS_FULL_RUN_DONE"
