#!/usr/bin/env bash
# DCE-V1 unattended runner; writes only DCE_OUTPUT. Never removes historical files.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
ROOT=/home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python
export DCE_OUTPUT="${DCE_OUTPUT:-$ROOT/outputs/care_dce_v1}"
DEVICE="${DCE_DEVICE:-cuda}"
RESUME=()
if [[ "${DCE_RESUME:-0}" == 1 ]]; then RESUME=(--resume); fi
mkdir -p "$DCE_OUTPUT"
exec > >(tee -a "$DCE_OUTPUT/run.log") 2>&1
echo "DCE_RUN_START output=$DCE_OUTPUT device=$DEVICE"
for phase in audit-only prepare-directions train-source source-eval train-final2017 evaluate-future collect; do
  marker="$DCE_OUTPUT/phases/$phase.done"
  if [[ -e "$marker" ]]; then
    echo "DCE_PHASE_RESUME name=$phase"
    continue
  fi
  echo "DCE_PHASE_START name=$phase epoch=$(date +%s)"
  "$PY" "$ROOT/scripts/care-dce-v1/main.py" "--$phase" --device "$DEVICE" --workers 1 "${RESUME[@]}"
  mkdir -p "$DCE_OUTPUT/phases"
  printf 'DONE\n' > "$marker"
  echo "DCE_PHASE_DONE name=$phase epoch=$(date +%s)"
  RESUME=(--resume)
done
echo "DCE_RUN_COMPLETE output=$DCE_OUTPUT"
