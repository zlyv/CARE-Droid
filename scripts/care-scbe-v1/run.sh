#!/usr/bin/env bash
# SCBE isolated phase runner; never modifies historical CARE-Droid assets.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
ROOT=/home/boyuanjie/projects/tasa-droid
OUT="$ROOT/outputs/care_scbe_v1"
HERE="$ROOT/scripts/care-scbe-v1"
PY=/home/boyuanjie/envs/tif/bin/python
PHASE=${1:?choose --audit-only, --smoke-test, --train-source, --train-rolling, --evaluate, --ablation or --collect}
shift
DEVICE=cpu
if [[ "$PHASE" != --audit-only ]]; then
  for arg in "$@"; do
    if [[ "$arg" == cuda ]]; then DEVICE=cuda; fi
  done
fi
cd "$ROOT"
if [[ "$PHASE" == --audit-only ]]; then
  "$PY" "$HERE/preflight.py" --verify
  exit
fi
if [[ ! -f "$OUT/asset_manifest_before.json" ]]; then
  echo 'SCBE preflight missing; refusing task' >&2;exit 2
fi
"$PY" "$HERE/preflight.py" --verify
case "$PHASE" in
  --smoke-test|--train-source|--train-rolling)
    "$PY" "$HERE/train.py" "$PHASE" "$@"
    ;;
  --evaluate|--ablation|--collect)
    "$PY" "$HERE/evaluate.py" "$PHASE" "$@"
    ;;
  *) echo "Unknown phase: $PHASE" >&2; exit 2;;
esac
