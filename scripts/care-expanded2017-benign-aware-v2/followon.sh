#!/usr/bin/env bash
# Await the already-running isolated extraction session, then execute frozen V2 phases.
set -euo pipefail
PROJECT=/home/boyuanjie/projects/tasa-droid
SCRIPT="$PROJECT/scripts/care-expanded2017-benign-aware-v2"
OUT="$PROJECT/outputs/care_expanded2017_benign_aware_v2"
PYTHON=/home/boyuanjie/envs/tif/bin/python
export EXPANDED_OUTPUT="$OUT"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$OUT/tmp"
export XDG_CACHE_HOME="$OUT/cache_home"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"

phase() {
  local name="$1"
  shift
  echo "EXPANDED_PHASE_START $name $(date -u +%FT%TZ)"
  "$@"
  echo "EXPANDED_PHASE_DONE $name $(date -u +%FT%TZ)"
}

echo "EXPANDED_FOLLOWON_WAITING_FOR_EXTRACT $(date -u +%FT%TZ)"
while tmux has-session -t care-expanded2017-v2-extract 2>/dev/null; do
  sleep 60
done
echo "EXPANDED_FOLLOWON_EXTRACTION_SESSION_EXITED $(date -u +%FT%TZ)"
if ! grep -q 'P31_BATCH_DONE' "$OUT/extraction.console.log"; then
  echo 'Extraction failed before the full P31 batch completed' >&2
  exit 1
fi
"$PYTHON" -c 'import json,pathlib; p=pathlib.Path("/home/boyuanjie/projects/tasa-droid/outputs/care_expanded2017_benign_aware_v2/stage2_union_new_2017/builder_status.json"); a=json.loads(p.read_text()); assert a["compact_rc"]==0 and a["endpoint_rc"]==0, a'
phase stage1 "$PYTHON" "$SCRIPT/train_stage1.py" --train-stage1 --resume --workers 2
phase router "$PYTHON" "$SCRIPT/train_router.py" --train-router --resume --workers 2
phase routed_deep_extraction "$PYTHON" "$SCRIPT/extract_features.py" --extract-stage2-routed --resume --workers 1
"$PYTHON" -c 'import json,pathlib; p=pathlib.Path("/home/boyuanjie/projects/tasa-droid/outputs/care_expanded2017_benign_aware_v2/stage2_routed_new_future/builder_status.json"); a=json.loads(p.read_text()); assert a["compact_rc"]==0 and a["endpoint_rc"]==0, a'
phase stage2_data "$PYTHON" "$SCRIPT/build_stage2_data.py" --build-stage2-data --resume

DEVICE=cpu
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_FREE=$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || true)
  GPU_UTIL=$(nvidia-smi --id=1 --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || true)
  if [[ "$GPU_FREE" =~ ^[0-9]+$ && "$GPU_UTIL" =~ ^[0-9]+$ ]] &&
     (( GPU_FREE >= 4096 && GPU_UTIL <= 20 )); then
    DEVICE=cuda:1
  fi
fi
echo "EXPANDED_STAGE2_DEVICE $DEVICE"
phase stage2_training "$PYTHON" "$SCRIPT/train_stage2.py" --train-stage2 --resume --device "$DEVICE"
phase evaluation "$PYTHON" "$SCRIPT/evaluate_cascade.py" --evaluate
phase report "$PYTHON" "$SCRIPT/collect_results.py" --collect
echo "CARE_EXPANDED2017_BENIGN_AWARE_V2_PIPELINE_DONE decision=$OUT/final_decision.json"
