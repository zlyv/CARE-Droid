#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python
S=scripts/care-droid-formal-diagnostics-v1
A=artifacts/care-droid-formal-diagnostics-v1
mkdir -p "$A/logs"
export PYTHONWARNINGS=ignore
export LOGURU_LEVEL=ERROR
export TQDM_DISABLE=1

wait_gpu() {
  local gpu="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then return 0; fi
  while true; do
    local active
    active="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
    if [[ -z "$active" ]]; then break; fi
    echo "WAITING_FOR_GPU${gpu}_IDLE active_pids=$(echo "$active" | tr '\n' ',')"
    sleep 60
  done
}

"$PY" "$S/selftest.py"
"$PY" "$S/preflight.py"
"$PY" "$S/build_references.py"
"$PY" "$S/build_a0_cache.py"
"$PY" "$S/prepare_future_s7.py"

if [[ ! -s "$A/frozen/future_predictions_label_free.csv" ]]; then
  wait_gpu 0
  (CUDA_VISIBLE_DEVICES=0 "$PY" "$S/run_frozen_future.py" --device cuda >"$A/logs/frozen.gpu0.log" 2>&1) &
  PID_FROZEN=$!
else
  echo "[resume] frozen future predictions exist"
  PID_FROZEN=""
fi

if [[ ! -s "$A/within_2018_2019/predictions.csv" ]]; then
  wait_gpu 1
  (CUDA_VISIBLE_DEVICES=1 "$PY" "$S/run_within_year.py" --device cuda --years 2018 2019 >"$A/logs/within_2018_2019.gpu1.log" 2>&1) &
  PID_WITHIN1=$!
else
  echo "[resume] within 2018/2019 predictions exist"
  PID_WITHIN1=""
fi

if [[ -n "$PID_FROZEN" ]]; then wait "$PID_FROZEN"; fi
echo "[done] frozen/GPU0"
if [[ ! -s "$A/within_2020_2021/predictions.csv" ]]; then
  wait_gpu 0
  CUDA_VISIBLE_DEVICES=0 "$PY" "$S/run_within_year.py" --device cuda --years 2020 2021 >"$A/logs/within_2020_2021.gpu0.log" 2>&1
else
  echo "[resume] within 2020/2021 predictions exist"
fi
if [[ -n "$PID_WITHIN1" ]]; then wait "$PID_WITHIN1"; fi
echo "[done] within-year groups"
"$PY" "$S/summarize.py"
echo CARE_DROID_FORMAL_DIAGNOSTICS_PIPELINE_DONE
echo artifacts=$A
echo report=$A/REPORT.txt
echo final_summary=$A/final_summary.json
