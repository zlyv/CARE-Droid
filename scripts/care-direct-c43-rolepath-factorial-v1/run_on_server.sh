#!/usr/bin/env bash
set -euo pipefail
PROJECT="${RPF_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${RPF_OUTPUT:-$PROJECT/outputs/care-direct-c43-rolepath-factorial-v1}"
PYTHON="${RPF_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
GPU="${RPF_GPU:-1}"
case "$OUTPUT" in "$PROJECT"/outputs/care-direct-c43-rolepath-factorial-v1);; *) echo "BLOCKED_BY_CONTRACT: unsafe output";exit 1;; esac
test -x "$PYTHON" || { echo "BLOCKED_BY_CONTRACT: Python missing";exit 1; }
mkdir -p "$OUTPUT/logs";export RPF_PROJECT="$PROJECT" RPF_OUTPUT="$OUTPUT" PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}";cd "$PROJECT"
phase(){ local name="$1";shift;echo "RPF_PHASE_START name=$name epoch=$(date +%s)";"$@";echo "RPF_PHASE_DONE name=$name epoch=$(date +%s)"; }
echo "CARE_DIRECT_C43_ROLEPATH_FACTORIAL_START output=$OUTPUT gpu=$GPU"
phase syntax "$PYTHON" -m compileall -q "$SCRIPT_DIR"
phase selftest "$PYTHON" "$SCRIPT_DIR/selftest.py"
phase preflight "$PYTHON" "$SCRIPT_DIR/preflight.py"
for arm in R11_ROLE_QUERY_READOUT R10_ROLE_QUERY_ONLY R01_ROLE_READOUT_ONLY R00_ROLE_NEITHER;do
  echo "RPF_ARM_START arm=$arm gpu=$GPU"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$SCRIPT_DIR/run_arm.py" --arm "$arm" --device cuda 2>&1 | tee "$OUTPUT/logs/$arm.gpu$GPU.log"
  echo "RPF_ARM_DONE arm=$arm"
done
phase summary "$PYTHON" "$SCRIPT_DIR/summarize.py"
phase integrity "$PYTHON" "$SCRIPT_DIR/integrity_after.py"
phase report "$PYTHON" "$SCRIPT_DIR/make_report.py"
echo "report=$OUTPUT/REPORT.md";echo "decision=$OUTPUT/final_decision.json"
echo "CARE_DIRECT_C43_ROLEPATH_FACTORIAL_DONE"
