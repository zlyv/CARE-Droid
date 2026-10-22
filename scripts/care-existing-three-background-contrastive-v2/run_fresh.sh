#!/usr/bin/env bash
# Repeat the complete protocol into a never-before-used versioned output dir.
set -euo pipefail

TASK_ROOT=/home/boyuanjie/projects/tasa-droid
TASK_SCRIPT="$TASK_ROOT/scripts/care-existing-three-background-contrastive-v2"
TASK_PYTHON=/home/boyuanjie/envs/tif/bin/python
TASK_OUTPUT="${1:?pass a new absolute output directory, e.g. ..._v3}"
case "$TASK_OUTPUT" in
  "$TASK_ROOT"/outputs/care_existing_three_background_contrastive_v[0-9]*) ;;
  *) echo 'Unsafe output path' >&2; exit 2 ;;
esac
if test -e "$TASK_OUTPUT"; then
  echo "Output already exists; refusing to overwrite: $TASK_OUTPUT" >&2
  exit 2
fi
mkdir "$TASK_OUTPUT"
export ETBC_OUTPUT="$TASK_OUTPUT"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd "$TASK_ROOT"

"$TASK_PYTHON" -B "$TASK_SCRIPT/audit.py"
"$TASK_PYTHON" -B "$TASK_SCRIPT/diagnose.py"
for TASK_ARM in M0 M1 M2 M3 M4 M5; do
  "$TASK_PYTHON" -B "$TASK_SCRIPT/contrastive_training.py" \
    --pilot --tag T2_C1 --seed 900000 --arm "$TASK_ARM" --device cuda
done

for TASK_TAG in T2_C1 T3_C1 T4_C1; do
  for TASK_SEED in 900000 900001 900002; do
    TASK_DIR="$TASK_OUTPUT/full/$TASK_TAG/$TASK_SEED"
    mkdir -p "$TASK_DIR"
    "$TASK_PYTHON" -B "$TASK_SCRIPT/contrastive_training.py" \
      --tag "$TASK_TAG" --seed "$TASK_SEED" --arm M0 --device cuda \
      > "$TASK_DIR/M0.log" 2>&1
    "$TASK_PYTHON" -B "$TASK_SCRIPT/verify_m0.py" "$TASK_TAG" "$TASK_SEED"
    for TASK_ARM in M1 M2 M3 M4 M5; do
      "$TASK_PYTHON" -B "$TASK_SCRIPT/contrastive_training.py" \
        --tag "$TASK_TAG" --seed "$TASK_SEED" --arm "$TASK_ARM" \
        --device cuda > "$TASK_DIR/${TASK_ARM}.log" 2>&1
    done
  done
done

"$TASK_PYTHON" -B "$TASK_SCRIPT/summarize_corrected.py"
"$TASK_PYTHON" -B "$TASK_SCRIPT/geometry.py"
"$TASK_PYTHON" -B "$TASK_SCRIPT/finalize_corrected.py"
