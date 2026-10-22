#!/usr/bin/env bash
set -euo pipefail
PROJECT="${CCBC_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="${CCBC_OUTPUT:-$PROJECT/outputs/care-ccbc-diagnostics-v1}"
SOURCE_OUT="${CCBC_SOURCE_OUTPUT:-$PROJECT/outputs/care-ccbc-source-v3}"
mkdir -p "$OUT/logs" "$OUT/checkpoints"

if [[ -n "${CCBC_PYTHON:-}" ]]; then PY="$CCBC_PYTHON"
elif [[ -x /home/boyuanjie/envs/tif/bin/python ]]; then PY=/home/boyuanjie/envs/tif/bin/python
elif command -v python3 >/dev/null 2>&1; then PY="$(command -v python3)"
else echo "BLOCKED_BY_CONTRACT: no Python interpreter" >&2; exit 1; fi

export CCBC_PROJECT="$PROJECT" CCBC_OUTPUT="$OUT" CCBC_SOURCE_OUTPUT="$SOURCE_OUT"
export CCBC_FUTURE_DIAGNOSTIC_ACK=1 CCBC_PYTHON="$PY"
export PYTHONWARNINGS=ignore LOGURU_LEVEL=ERROR TQDM_DISABLE=1 CCBC_JOBS="${CCBC_JOBS:-4}"
cd "$PROJECT"

phase() {
  local name="$1"; shift
  local marker="$OUT/checkpoints/$name.done"
  if [[ -s "$marker" ]]; then echo "CCBC_DIAGNOSTIC_PHASE_RESUME name=$name"; return 0; fi
  local started; started="$(date +%s)"; echo "CCBC_DIAGNOSTIC_PHASE_START name=$name epoch=$started"
  "$@"
  echo "ok" > "$marker"
  echo "CCBC_DIAGNOSTIC_PHASE_DONE name=$name wall_seconds=$(($(date +%s)-started))"
}

wait_gpu() {
  local gpu="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then return 0; fi
  while true; do
    local active; active="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
    [[ -z "$active" ]] && return 0
    echo "CCBC_WAITING_FOR_GPU${gpu}_IDLE active_pids=$(echo "$active" | tr '\n' ',')"; sleep 60
  done
}

echo "CARE_CCBC_DIAGNOSTICS_PIPELINE_START output=$OUT source=$SOURCE_OUT python=$PY"
phase syntax "$PY" -m py_compile \
  "$SCRIPT_DIR"/common.py "$SCRIPT_DIR"/contracts.py "$SCRIPT_DIR"/ccbc_evidence.py \
  "$SCRIPT_DIR"/diagnostic_assets.py "$SCRIPT_DIR"/prepare_future_scope.py \
  "$SCRIPT_DIR"/replay_diagnostic_attention.py "$SCRIPT_DIR"/run_frozen_diagnostic.py \
  "$SCRIPT_DIR"/run_within_diagnostic.py "$SCRIPT_DIR"/summarize_diagnostics.py
phase selftest "$PY" "$SCRIPT_DIR/selftest.py"
phase preflight "$PY" "$SCRIPT_DIR/preflight_diagnostics.py"
phase future_scope "$PY" "$SCRIPT_DIR/prepare_future_scope.py"

GPU_COUNT=0
if command -v nvidia-smi >/dev/null 2>&1; then GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l)"; fi
GPU_FROZEN="${CCBC_GPU_FROZEN:-0}"
GPU_WITHIN="${CCBC_GPU_WITHIN:-1}"
for gpu in "$GPU_FROZEN" "$GPU_WITHIN"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ || "$gpu" -ge "$GPU_COUNT" ]]; then
    echo "BLOCKED_BY_CONTRACT: requested GPU $gpu but detected GPU_COUNT=$GPU_COUNT" >&2
    exit 1
  fi
done

if [[ "$GPU_FROZEN" == "$GPU_WITHIN" ]]; then
  echo "CCBC_GPU_PLAN frozen=$GPU_FROZEN within=$GPU_WITHIN mode=sequential"
  if [[ ! -s "$OUT/cache/future_attention_frozen.npz" ]]; then
    wait_gpu "$GPU_FROZEN"
    CUDA_VISIBLE_DEVICES="$GPU_FROZEN" "$PY" "$SCRIPT_DIR/replay_diagnostic_attention.py" frozen --device cuda \
      > >(tee "$OUT/logs/frozen_attention.log") 2>&1
  else
    echo "CCBC_FROZEN_ATTENTION_RESUME_PASS"
  fi
  echo "CCBC_FROZEN_ATTENTION_READY"

  if [[ ! -s "$OUT/cache/future_attention_within.npz" ]]; then
    wait_gpu "$GPU_WITHIN"
    CUDA_VISIBLE_DEVICES="$GPU_WITHIN" "$PY" "$SCRIPT_DIR/replay_diagnostic_attention.py" within --device cuda \
      > >(tee "$OUT/logs/within_attention.log") 2>&1
  else
    echo "CCBC_WITHIN_ATTENTION_RESUME_PASS"
  fi
  echo "CCBC_WITHIN_ATTENTION_READY"
else
  echo "CCBC_GPU_PLAN frozen=$GPU_FROZEN within=$GPU_WITHIN mode=parallel_when_idle"
  if [[ ! -s "$OUT/cache/future_attention_frozen.npz" ]]; then
    (wait_gpu "$GPU_FROZEN"; CUDA_VISIBLE_DEVICES="$GPU_FROZEN" \
      "$PY" "$SCRIPT_DIR/replay_diagnostic_attention.py" frozen --device cuda \
      > >(tee "$OUT/logs/frozen_attention.log") 2>&1) & PID_FROZEN=$!
  else
    echo "CCBC_FROZEN_ATTENTION_RESUME_PASS"; PID_FROZEN=""
  fi

  if [[ ! -s "$OUT/cache/future_attention_within.npz" ]]; then
    (wait_gpu "$GPU_WITHIN"; CUDA_VISIBLE_DEVICES="$GPU_WITHIN" \
      "$PY" "$SCRIPT_DIR/replay_diagnostic_attention.py" within --device cuda \
      > >(tee "$OUT/logs/within_attention.log") 2>&1) & PID_WITHIN=$!
  else
    echo "CCBC_WITHIN_ATTENTION_RESUME_PASS"; PID_WITHIN=""
  fi

  [[ -z "$PID_FROZEN" ]] || wait "$PID_FROZEN"
  echo "CCBC_FROZEN_ATTENTION_READY"
  [[ -z "$PID_WITHIN" ]] || wait "$PID_WITHIN"
  echo "CCBC_WITHIN_ATTENTION_READY"
fi

if [[ ! -s "$OUT/frozen_diagnostic.json" ]]; then
  ("$PY" "$SCRIPT_DIR/run_frozen_diagnostic.py" > >(tee "$OUT/logs/frozen_ccbc.log") 2>&1) & PID_FD=$!
else echo "CCBC_FROZEN_DIAGNOSTIC_RESUME_PASS"; PID_FD=""; fi
if [[ ! -s "$OUT/within_diagnostic.json" ]]; then
  ("$PY" "$SCRIPT_DIR/run_within_diagnostic.py" > >(tee "$OUT/logs/within_ccbc.log") 2>&1) & PID_WD=$!
else echo "CCBC_WITHIN_DIAGNOSTIC_RESUME_PASS"; PID_WD=""; fi
[[ -z "$PID_FD" ]] || wait "$PID_FD"
[[ -z "$PID_WD" ]] || wait "$PID_WD"
phase integrity "$PY" "$SCRIPT_DIR/integrity_diagnostics.py"
phase summary "$PY" "$SCRIPT_DIR/summarize_diagnostics.py"
echo CARE_CCBC_DIAGNOSTICS_PIPELINE_DONE
echo output="$OUT"
echo report="$OUT/REPORT.txt"
echo final_summary="$OUT/final_diagnostic_summary.json"
