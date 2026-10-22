#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python
SCRIPT=scripts/care-p31rf-c43-p0-complete-diagnostics-v1
ROOT=artifacts/care-p31rf-c43-p0-complete-diagnostics-v1
LOGS="$ROOT/logs"
mkdir -p "$LOGS" "$ROOT/frozen_future/model" "$ROOT/within_year/2018_2019" "$ROOT/within_year/2020_2021" "$ROOT/results"
"$PY" "$SCRIPT/selftest.py"
"$PY" "$SCRIPT/preflight.py"
run_frozen(){ CUDA_VISIBLE_DEVICES=0 "$PY" "$SCRIPT/run_frozen_future.py" --representation anomaly --head bounded --device cuda --seeds 900000,900001,900002 --max-delta 2.25 --output "$ROOT/frozen_future/model"; }
run_within(){ local gpu="$1" years="$2" name="$3"; CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$SCRIPT/run_within_year.py" --representation anomaly --head bounded --device cuda --seeds 900000,900001,900002 --max-delta 2.25 --years "$years" --output "$ROOT/within_year/$name"; }
echo "[run] frozen 2017-to-future on GPU0"
run_frozen >"$LOGS/frozen.gpu0.log" 2>&1 & pf=$!
echo "[run] same-year 2018,2019 on GPU1"
run_within 1 2018,2019 2018_2019 >"$LOGS/within_2018_2019.gpu1.log" 2>&1 & pe=$!
status=0
wait "$pf" || status=1
if [[ "$status" -eq 0 ]];then echo "[done] frozen/GPU0";echo "[run] same-year 2020,2021 on GPU0";run_within 0 2020,2021 2020_2021 >"$LOGS/within_2020_2021.gpu0.log" 2>&1 & pl=$!;else pl="";fi
wait "$pe" || status=1
if [[ -n "$pl" ]];then wait "$pl" || status=1;fi
if [[ "$status" -ne 0 ]];then echo "BLOCKED_BY_CONTRACT: inspect $LOGS";exit 1;fi
"$PY" "$SCRIPT/summarize.py"
echo P31RF_C43_P0_COMPLETE_FULL_RUN_DONE
