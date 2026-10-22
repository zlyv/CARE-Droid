#!/usr/bin/env bash
# Launch DCE-V1 once in a detached tmux session; no historical assets changed.
set -euo pipefail
ROOT=/home/boyuanjie/projects/tasa-droid
SESSION="${DCE_SESSION:-care-dce-v1}"
OUT="${DCE_OUTPUT:-$ROOT/outputs/care_dce_v1}"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Already running: $SESSION"
  exit 0
fi
if [[ -e "$OUT" && ! -e "$OUT/asset_manifest_before.json" ]]; then
  echo "Output occupied without manifest; refusing to overwrite: $OUT" >&2
  exit 1
fi
if [[ -e "$OUT/final_decision.json" ]]; then
  echo "Experiment already complete: $OUT" >&2
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export DCE_DEVICE="${DCE_DEVICE:-cuda}"
export DCE_OUTPUT="$OUT"
RESUME=0
if [[ -e "$OUT/asset_manifest_before.json" ]]; then RESUME=1; fi
mkdir -p "$OUT"
tmux new-session -d -s "$SESSION" \
  "env DCE_OUTPUT='$OUT' DCE_DEVICE='$DCE_DEVICE' DCE_RESUME='$RESUME' CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' PYTHONDONTWRITEBYTECODE=1 bash '$ROOT/scripts/care-dce-v1/run.sh'"
printf 'session=%s\noutput=%s\ndevice=%s\ncuda_visible_devices=%s\n' "$SESSION" "$OUT" "$DCE_DEVICE" "$CUDA_VISIBLE_DEVICES" > "$OUT/launch_info.txt"
echo "Started: $SESSION"
echo "Monitor: tail -F $OUT/run.log"
echo "Attach: tmux attach -t $SESSION"
