#!/usr/bin/env bash
# HDBC-V1 one-shot detached launcher; no other tmux sessions touched.
set -euo pipefail
ROOT=/home/boyuanjie/projects/tasa-droid
SESSION="${HDBC_SESSION:-care_hdbc_v1_$(date +%Y%m%d_%H%M%S)}"
OUT="${HDBC_OUTPUT:-$ROOT/outputs/care_hdbc_v1}"
DEVICE="${HDBC_DEVICE:-cuda}"
GPU="${CUDA_VISIBLE_DEVICES:-1}"
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "Already running: $SESSION"; exit 0; fi
if [[ -e "$OUT/final_decision.json" ]]; then echo "Already complete: $OUT" >&2; exit 1; fi
if [[ -e "$OUT" && ! -e "$OUT/asset_manifest_before.json" ]]; then echo "Output occupied without HDBC manifest: $OUT" >&2; exit 1; fi
resume=0
if [[ -e "$OUT/asset_manifest_before.json" ]]; then resume=1; fi
mkdir -p "$OUT"
tmux new-session -d -s "$SESSION" \
  "env HDBC_OUTPUT='$OUT' HDBC_DEVICE='$DEVICE' HDBC_RESUME='$resume' CUDA_VISIBLE_DEVICES='$GPU' PYTHONDONTWRITEBYTECODE=1 bash '$ROOT/scripts/care-hdbc-v1/run.sh'"
printf 'session=%s\noutput=%s\ndevice=%s\ngpu=%s\n' "$SESSION" "$OUT" "$DEVICE" "$GPU" > "$OUT/launch_info.txt"
echo "Started: $SESSION"
echo "Monitor: tail -F $OUT/run.log"
