#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid

PY="${CSVTW_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
S="scripts/care-svm-train-window-expansion-v1"
O="outputs/care-svm-train-window-expansion-v1"
GPU="${CSVTW_GPU:-1}"
mkdir -p "$O/logs"
export CSVTW_PROJECT=/home/boyuanjie/projects/tasa-droid
export CSVTW_OUTPUT=/home/boyuanjie/projects/tasa-droid/outputs/care-svm-train-window-expansion-v1
export PYTHONWARNINGS=ignore LOGURU_LEVEL=ERROR TQDM_DISABLE=1

phase() {
  local name="$1"; shift
  if [[ -f "$O/checkpoints/${name}.done" ]]; then
    echo "CSVTW_PHASE_RESUME name=$name"; return 0
  fi
  echo "CSVTW_PHASE_START name=$name epoch=$(date +%s)"
  "$@"
  mkdir -p "$O/checkpoints"; touch "$O/checkpoints/${name}.done"
  echo "CSVTW_PHASE_DONE name=$name epoch=$(date +%s)"
}

wait_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then return 0; fi
  while true; do
    active="$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
    if [[ -z "$active" ]]; then return 0; fi
    echo "CSVTW_WAITING_FOR_GPU${GPU}_IDLE active_pids=$(echo "$active" | tr '\n' ',')"; sleep 60
  done
}

echo "CARE_SVM_TRAIN_WINDOW_EXPANSION_START output=$O gpu=$GPU"
phase syntax "$PY" -m py_compile "$S"/*.py
phase selftest "$PY" "$S/selftest.py"
phase preflight "$PY" "$S/preflight.py"
wait_gpu
phase W18 env CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$S/run_window.py" --window W18 --device cuda
phase W1718 env CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$S/run_window.py" --window W1718 --device cuda
phase evaluation "$PY" "$S/evaluate.py"
phase summary "$PY" "$S/summarize.py"
echo "CARE_SVM_TRAIN_WINDOW_EXPANSION_DONE"
echo "report=$O/REPORT.md"
echo "results=$O/train_window_results.csv"
echo "decision=$O/final_decision.json"
