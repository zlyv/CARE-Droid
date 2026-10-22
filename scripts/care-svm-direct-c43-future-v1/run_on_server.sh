#!/usr/bin/env bash
set -euo pipefail
PROJECT="${SDC43F_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${SDC43F_OUTPUT:-$PROJECT/outputs/care-svm-direct-c43-future-v1}"
PYTHON="${SDC43F_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
GPU="${SDC43F_GPU:-1}"
case "$OUTPUT" in "$PROJECT"/outputs/care-svm-direct-c43-future-v1);; *) echo "BLOCKED_BY_CONTRACT: unsafe output"; exit 1;; esac
test -x "$PYTHON" || { echo "BLOCKED_BY_CONTRACT: Python missing"; exit 1; }
mkdir -p "$OUTPUT/logs" "$OUTPUT/checkpoints"
export SDC43F_PROJECT="$PROJECT" SDC43F_OUTPUT="$OUTPUT" PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONWARNINGS=ignore LOGURU_LEVEL=ERROR TQDM_DISABLE=1
cd "$PROJECT"
phase(){ local name="$1"; shift; echo "SDC43F_PHASE_START name=$name epoch=$(date +%s)"; "$@"; echo "SDC43F_PHASE_DONE name=$name epoch=$(date +%s)"; }
echo "CARE_SVM_DIRECT_C43_FUTURE_START output=$OUTPUT gpu=$GPU"
phase syntax "$PYTHON" -m compileall -q "$SCRIPT_DIR"
phase selftest "$PYTHON" "$SCRIPT_DIR/selftest.py"
phase preflight "$PYTHON" "$SCRIPT_DIR/preflight.py"
echo "SDC43F_FUTURE_START gpu=$GPU labels=false"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$SCRIPT_DIR/run_future.py" --device cuda 2>&1 | tee "$OUTPUT/logs/future.gpu$GPU.log"
echo "SDC43F_FUTURE_DONE labels=false"
phase summary "$PYTHON" "$SCRIPT_DIR/summarize.py"
echo "CARE_SVM_DIRECT_C43_FUTURE_PIPELINE_DONE"
echo "report=$OUTPUT/REPORT.md"
echo "decision=$OUTPUT/final_decision.json"
