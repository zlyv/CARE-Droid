#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid

PY="${PLRRS_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
S="scripts/care-p31-lr-robust-static-v1"
O="outputs/care-p31-lr-robust-static-v1"
GPU_TEXT="${PLRRS_GPUS:-1}"
mkdir -p "$O/logs" "$O/checkpoints"
export PCFS2_PROJECT=/home/boyuanjie/projects/tasa-droid
export PCFS2_OUTPUT=/home/boyuanjie/projects/tasa-droid/outputs/care-p31-lr-robust-static-v1
export PYTHONWARNINGS=ignore
export LOGURU_LEVEL=ERROR
export TQDM_DISABLE=1

phase() {
  local name="$1"; shift
  echo "PLRRS_PHASE_START name=$name epoch=$(date +%s)"
  "$@"
  echo "PLRRS_PHASE_DONE name=$name epoch=$(date +%s)"
}

wait_gpu() {
  local gpu="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then return 0; fi
  while true; do
    local active
    active="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
    if [[ -z "$active" ]]; then return 0; fi
    echo "PLRRS_WAITING_FOR_GPU${gpu}_IDLE active_pids=$(echo "$active" | tr '\n' ',')"
    sleep 60
  done
}

IFS=',' read -r -a GPUS <<< "$GPU_TEXT"
if [[ "${#GPUS[@]}" -lt 1 ]]; then echo "PLRRS_GPUS is empty"; exit 2; fi
ARMS=(L1_HELLINGER L2_LOG L3_PRESENCE L4_RAW_PRESENCE L5_CLR_PRESENCE L6_HELLINGER_PRESENCE)

run_lanes() {
  local phase_name="$1" script="$2"
  local pids=()
  for lane in "${!GPUS[@]}"; do
    local gpu="${GPUS[$lane]}"
    (
      wait_gpu "$gpu"
      for index in "${!ARMS[@]}"; do
        if (( index % ${#GPUS[@]} != lane )); then continue; fi
        local arm="${ARMS[$index]}" log="$O/logs/${phase_name}_${ARMS[$index]}.gpu${gpu}.log"
        echo "PLRRS_${phase_name^^}_START arm=$arm gpu=$gpu log=$log"
        CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$S/$script" --arm "$arm" --device cuda >"$log" 2>&1
        echo "PLRRS_${phase_name^^}_DONE arm=$arm gpu=$gpu"
      done
    ) &
    pids+=("$!")
  done
  local pid
  for pid in "${pids[@]}"; do wait "$pid"; done
}

echo "CARE_P31_LR_ROBUST_STATIC_PIPELINE_START output=$O python=$PY gpus=$GPU_TEXT"
phase syntax "$PY" -m py_compile "$S"/*.py
phase selftest "$PY" "$S/selftest.py"
phase preflight "$PY" "$S/preflight.py"
phase source_references "$PY" "$S/build_source_references.py"

echo "PLRRS_SOURCE_ALL_START"
run_lanes source run_source_arm.py
echo "PLRRS_SOURCE_ALL_DONE"
phase source_summary "$PY" "$S/summarize_source.py"

echo "PLRRS_FROZEN_ALL_START"
run_lanes frozen run_frozen_arm.py
echo "PLRRS_FROZEN_ALL_DONE"
phase drift_audit "$PY" "$S/drift_audit.py"

echo "PLRRS_WITHIN_ALL_START"
run_lanes within run_within_arm.py
echo "PLRRS_WITHIN_ALL_DONE"

phase final_summary "$PY" "$S/summarize_all.py"
echo "CARE_P31_LR_ROBUST_STATIC_PIPELINE_DONE"
echo "report=$O/REPORT.md"
echo "table=$O/full_system_comparison.csv"
echo "decision=$O/final_decision.json"
