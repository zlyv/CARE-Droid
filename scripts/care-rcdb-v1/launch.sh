#!/usr/bin/env bash
# RCDB-V1 launcher; creates only a new output and tmux session, never touches historical assets.
set -Eeuo pipefail
ROOT=/home/boyuanjie/projects/tasa-droid
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/outputs/care_rcdb_v1"
if [ -e "$OUT" ]; then
  OUT="$ROOT/outputs/care_rcdb_v1_$STAMP"
fi
if [ -e "$OUT" ]; then echo "No unused output path: $OUT" >&2; exit 1; fi
SESSION="care_rcdb_v1_$STAMP"
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "Session exists" >&2; exit 1; fi
DEVICE="${RCDB_DEVICE:-cuda}"
if [ "$DEVICE" = cuda ]; then
  FREE="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${CUDA_VISIBLE_DEVICES:-0}" | head -n 1 | tr -d ' ')"
  if [ -z "$FREE" ] || [ "$FREE" -lt 4096 ]; then echo "GPU free memory below 4 GiB" >&2; exit 1; fi
fi
tmux new-session -d -s "$SESSION" "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-0}' RCDB_OUTPUT='$OUT' RCDB_DEVICE='$DEVICE' bash '$ROOT/scripts/care-rcdb-v1/run.sh'"
tmux has-session -t "$SESSION"
printf 'SESSION=%s\nOUTPUT=%s\nDEVICE=%s\n' "$SESSION" "$OUT" "$DEVICE"
