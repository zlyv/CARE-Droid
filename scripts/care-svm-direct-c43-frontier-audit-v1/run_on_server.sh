#!/usr/bin/env bash
set -euo pipefail
PROJECT="${DCFA_PROJECT:-/home/boyuanjie/projects/tasa-droid}";SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";OUTPUT="${DCFA_OUTPUT:-$PROJECT/outputs/care-svm-direct-c43-frontier-audit-v1}";PYTHON="${DCFA_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
export DCFA_PROJECT="$PROJECT" DCFA_OUTPUT="$OUTPUT";mkdir -p "$OUTPUT/checkpoints"
phase(){ local n="$1";shift;local m="$OUTPUT/checkpoints/$n.done";if [[ -f "$m" ]];then echo "DCFA_PHASE_RESUME name=$n";return;fi;echo "DCFA_PHASE_START name=$n";"$@";touch "$m";echo "DCFA_PHASE_DONE name=$n";}
echo "CARE_DIRECT_C43_FRONTIER_PIPELINE_START output=$OUTPUT"
phase syntax "$PYTHON" -m py_compile "$SCRIPT_DIR"/*.py
phase selftest "$PYTHON" "$SCRIPT_DIR/selftest.py"
phase preflight "$PYTHON" "$SCRIPT_DIR/preflight.py"
phase audit "$PYTHON" "$SCRIPT_DIR/run_audit.py"
phase integrity "$PYTHON" "$SCRIPT_DIR/integrity_after.py"
phase summary "$PYTHON" "$SCRIPT_DIR/summarize.py"
echo "CARE_DIRECT_C43_FRONTIER_PIPELINE_DONE report=$OUTPUT/report.md"
