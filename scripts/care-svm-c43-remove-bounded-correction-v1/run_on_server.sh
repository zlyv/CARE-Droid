#!/usr/bin/env bash
set -euo pipefail
PROJECT="${CSRC_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${CSRC_OUTPUT:-$PROJECT/outputs/care-svm-c43-remove-bounded-correction-v1}"
PYTHON="${CSRC_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
GPU="${CSRC_GPU:-1}"
case "$OUTPUT" in "$PROJECT"/outputs/care-svm-c43-remove-bounded-correction-v1);; *) echo "BLOCKED_BY_CONTRACT: unsafe output";exit 1;; esac
test -x "$PYTHON" || { echo "BLOCKED_BY_CONTRACT: Python missing";exit 1; }
mkdir -p "$OUTPUT/logs";export CSRC_PROJECT="$PROJECT" CSRC_OUTPUT="$OUTPUT" PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}";cd "$PROJECT"
phase(){ local name="$1";shift;echo "CSRC_PHASE_START name=$name epoch=$(date +%s)";"$@";echo "CSRC_PHASE_DONE name=$name epoch=$(date +%s)"; }
echo "CARE_SVM_C43_REMOVE_BOUNDED_START output=$OUTPUT gpu=$GPU"
phase syntax "$PYTHON" -m compileall -q "$SCRIPT_DIR"
phase selftest "$PYTHON" "$SCRIPT_DIR/selftest.py"
phase preflight "$PYTHON" "$SCRIPT_DIR/preflight.py"
for arm in U1_UNBOUNDED_RESIDUAL U2_DIRECT_STAGE2;do
  echo "CSRC_ARM_START arm=$arm gpu=$GPU"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$SCRIPT_DIR/run_arm.py" --arm "$arm" --device cuda 2>&1 | tee "$OUTPUT/logs/$arm.gpu$GPU.log"
  echo "CSRC_ARM_DONE arm=$arm"
done
phase summary "$PYTHON" "$SCRIPT_DIR/summarize.py"
phase integrity "$PYTHON" "$SCRIPT_DIR/integrity_after.py"
phase report "$PYTHON" "$SCRIPT_DIR/make_report.py"
echo "report=$OUTPUT/REPORT.md";echo "decision=$OUTPUT/final_decision.json"
