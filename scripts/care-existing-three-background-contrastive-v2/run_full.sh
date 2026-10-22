#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export ETBC_OUTPUT=/home/boyuanjie/projects/tasa-droid/outputs/care_existing_three_background_contrastive_v2
TASK_ROOT=/home/boyuanjie/projects/tasa-droid
TASK_SCRIPT="$TASK_ROOT/scripts/care-existing-three-background-contrastive-v2"
TASK_PYTHON=/home/boyuanjie/envs/tif/bin/python
cd "$TASK_ROOT"
test -f "$ETBC_OUTPUT/baseline_replay.json"
for TASK_TAG in T2_C1 T3_C1 T4_C1; do
  for TASK_SEED in 900000 900001 900002; do
    TASK_DIR="$ETBC_OUTPUT/full/$TASK_TAG/$TASK_SEED"
    mkdir -p "$TASK_DIR"
    if test -e "$TASK_DIR/M0_predictions.npz"; then
      echo "ETBC_EXISTING_M0_VERIFY $TASK_TAG $TASK_SEED"
    else
      test ! -e "$TASK_DIR/M0.log"
      "$TASK_PYTHON" -B "$TASK_SCRIPT/contrastive_training.py" --tag "$TASK_TAG" --seed "$TASK_SEED" --arm M0 --device cuda > "$TASK_DIR/M0.log" 2>&1
    fi
    if test ! -e "$TASK_DIR/M0_exact_retrain_check.json"; then
      "$TASK_PYTHON" -B "$TASK_SCRIPT/verify_m0.py" "$TASK_TAG" "$TASK_SEED"
    fi
    for TASK_ARM in M1 M2 M3 M4 M5; do
      test ! -e "$TASK_DIR/${TASK_ARM}_predictions.npz"
      test ! -e "$TASK_DIR/${TASK_ARM}.log"
      echo "ETBC_ARM_START $TASK_TAG $TASK_SEED $TASK_ARM"
      "$TASK_PYTHON" -B "$TASK_SCRIPT/contrastive_training.py" --tag "$TASK_TAG" --seed "$TASK_SEED" --arm "$TASK_ARM" --device cuda > "$TASK_DIR/${TASK_ARM}.log" 2>&1
      echo "ETBC_ARM_END $TASK_TAG $TASK_SEED $TASK_ARM"
    done
  done
done
echo ETBC_FULL_DONE
