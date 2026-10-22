#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid

PY="${PLRRS_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
S="scripts/care-p31-lr-robust-static-v1"
O="outputs/care-p31-lr-robust-static-l6-v1"
GPU="${PLRRS_GPU:-1}"
mkdir -p "$O/logs" "$O/checkpoints"
export PCFS2_PROJECT=/home/boyuanjie/projects/tasa-droid
export PCFS2_OUTPUT=/home/boyuanjie/projects/tasa-droid/outputs/care-p31-lr-robust-static-l6-v1
export PLRRS_ONLY_ARM=L6_HELLINGER_PRESENCE
export PYTHONWARNINGS=ignore LOGURU_LEVEL=ERROR TQDM_DISABLE=1

phase() {
  local name="$1"; shift
  echo "PLRRS_L6_PHASE_START name=$name epoch=$(date +%s)"
  "$@"
  echo "PLRRS_L6_PHASE_DONE name=$name epoch=$(date +%s)"
}

wait_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then return 0; fi
  while true; do
    local active
    active="$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
    if [[ -z "$active" ]]; then return 0; fi
    echo "PLRRS_L6_WAITING_FOR_GPU${GPU}_IDLE active_pids=$(echo "$active" | tr '\n' ',')"
    sleep 60
  done
}

echo "CARE_P31_LR_L6_ONLY_PIPELINE_START output=$O gpu=$GPU"
phase syntax "$PY" -m py_compile "$S"/*.py
phase selftest "$PY" "$S/selftest.py"
phase preflight "$PY" "$S/preflight.py"
phase source_references "$PY" "$S/build_source_references.py"

wait_gpu
echo "PLRRS_L6_SOURCE_START"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$S/run_source_arm.py" --arm L6_HELLINGER_PRESENCE --device cuda >"$O/logs/source_L6.gpu${GPU}.log" 2>&1
echo "PLRRS_L6_SOURCE_DONE"
phase source_summary "$PY" "$S/summarize_source.py"

wait_gpu
echo "PLRRS_L6_FROZEN_2018_2021_START"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$S/run_frozen_arm.py" --arm L6_HELLINGER_PRESENCE --device cuda >"$O/logs/frozen_L6.gpu${GPU}.log" 2>&1
echo "PLRRS_L6_FROZEN_2018_2021_DONE"
phase drift_audit "$PY" "$S/drift_audit.py"

wait_gpu
echo "PLRRS_L6_WITHIN_YEAR_START"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$S/run_within_arm.py" --arm L6_HELLINGER_PRESENCE --device cuda >"$O/logs/within_L6.gpu${GPU}.log" 2>&1
echo "PLRRS_L6_WITHIN_YEAR_DONE"

phase final_summary "$PY" "$S/summarize_all.py"
echo "CARE_P31_LR_L6_ONLY_PIPELINE_DONE"
echo "report=$O/REPORT.md"
echo "table=$O/full_system_comparison.csv"
echo "decision=$O/final_decision.json"
