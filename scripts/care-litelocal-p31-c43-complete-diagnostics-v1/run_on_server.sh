#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid

PY=/home/boyuanjie/envs/tif/bin/python
SCRIPT=scripts/care-litelocal-p31-c43-complete-diagnostics-v1
ROOT=artifacts/care-litelocal-p31-c43-complete-diagnostics-v1
LOGS="$ROOT/logs"
mkdir -p "$LOGS" "$ROOT/frozen_future/model" "$ROOT/within_year/2018_2019" "$ROOT/within_year/2020_2021" "$ROOT/results"

"$PY" "$SCRIPT/selftest.py"
"$PY" "$SCRIPT/preflight.py"

run_frozen() {
  CUDA_VISIBLE_DEVICES=0 "$PY" "$SCRIPT/run_frozen_future.py" \
    --representation anomaly --head bounded --device cuda \
    --seeds 900000,900001,900002 --max-delta 2.25 \
    --output "$ROOT/frozen_future/model"
}

run_within_group() {
  local gpu_id="$1"
  local years="$2"
  local output_name="$3"
  CUDA_VISIBLE_DEVICES="$gpu_id" "$PY" "$SCRIPT/run_within_year.py" \
    --representation anomaly --head bounded --device cuda \
    --seeds 900000,900001,900002 --max-delta 2.25 \
    --years "$years" --output "$ROOT/within_year/$output_name"
}

echo "[run] frozen 2017-to-future on GPU0"
run_frozen >"$LOGS/frozen.gpu0.log" 2>&1 &
pid_frozen=$!
echo "[run] same-year 2018,2019 on GPU1"
run_within_group 1 2018,2019 2018_2019 >"$LOGS/within_2018_2019.gpu1.log" 2>&1 &
pid_within_early=$!

status=0
wait "$pid_frozen" || status=1
if [[ "$status" -eq 0 ]]; then
  echo "[done] frozen/GPU0"
  echo "[run] same-year 2020,2021 on GPU0"
  run_within_group 0 2020,2021 2020_2021 >"$LOGS/within_2020_2021.gpu0.log" 2>&1 &
  pid_within_late=$!
else
  echo "[failed] frozen/GPU0"
  pid_within_late=""
fi

wait "$pid_within_early" || status=1
if [[ -n "$pid_within_late" ]]; then
  wait "$pid_within_late" || status=1
fi
if [[ "$status" -ne 0 ]]; then
  echo "BLOCKED_BY_CONTRACT: at least one complete-diagnostic job failed; inspect $LOGS"
  exit 1
fi

"$PY" "$SCRIPT/summarize.py"
echo "P31_C43_COMPLETE_DIAGNOSTICS_FULL_RUN_DONE"
