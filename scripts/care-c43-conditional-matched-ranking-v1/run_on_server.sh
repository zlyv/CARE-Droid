#!/usr/bin/env bash
set -euo pipefail
PROJECT="${CCMR_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${CCMR_OUTPUT:-$PROJECT/outputs/care-c43-conditional-matched-ranking-v1}"
PYTHON="${CCMR_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
export CCMR_PROJECT="$PROJECT" CCMR_OUTPUT="$OUTPUT" PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT/logs"
run_phase(){ local name="$1"; shift; local marker="$OUTPUT/checkpoints/${name}.done"; mkdir -p "$OUTPUT/checkpoints"; if [[ -f "$marker" ]]; then echo "CCMR_PHASE_RESUME name=$name"; return; fi; echo "CCMR_PHASE_START name=$name epoch=$(date +%s)"; "$@"; touch "$marker"; echo "CCMR_PHASE_DONE name=$name epoch=$(date +%s)"; }
echo "CARE_C43_CONDITIONAL_MATCHED_PIPELINE_START output=$OUTPUT python=$PYTHON"
run_phase syntax "$PYTHON" -m py_compile "$SCRIPT_DIR"/*.py
run_phase selftest "$PYTHON" "$SCRIPT_DIR/selftest.py"
run_phase preflight "$PYTHON" "$SCRIPT_DIR/preflight.py"
run_phase source_probe "$PYTHON" "$SCRIPT_DIR/run_probe.py"
run_phase integrity "$PYTHON" "$SCRIPT_DIR/integrity_after.py"
run_phase summary "$PYTHON" "$SCRIPT_DIR/summarize.py"
echo "CARE_C43_CONDITIONAL_MATCHED_PIPELINE_DONE report=$OUTPUT/report.md"
