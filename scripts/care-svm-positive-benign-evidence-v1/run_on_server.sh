#!/usr/bin/env bash
set -euo pipefail

PROJECT="${SVMPBE_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${SVMPBE_OUTPUT:-$PROJECT/outputs/care-svm-positive-benign-evidence-v1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SVMPBE_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"

mkdir -p "$OUTPUT/logs"
export SVMPBE_PROJECT="$PROJECT" SVMPBE_OUTPUT="$OUTPUT"
exec > >(tee -a "$OUTPUT/logs/run.log") 2>&1

phase() {
  local name="$1"; shift
  local marker="$OUTPUT/checkpoints/${name}.done"
  mkdir -p "$OUTPUT/checkpoints"
  if [[ -f "$marker" ]]; then echo "SVMPBE_PHASE_RESUME name=$name"; return; fi
  echo "SVMPBE_PHASE_START name=$name epoch=$(date +%s)"
  "$@"
  touch "$marker"
  echo "SVMPBE_PHASE_DONE name=$name epoch=$(date +%s)"
}

echo "CARE_SVM_POSITIVE_BENIGN_PIPELINE_START output=$OUTPUT python=$PYTHON"
phase syntax "$PYTHON" -m py_compile "$SCRIPT_DIR"/*.py
phase selftest "$PYTHON" "$SCRIPT_DIR/selftest.py"
phase preflight "$PYTHON" "$SCRIPT_DIR/preflight.py"
phase source "$PYTHON" "$SCRIPT_DIR/run_source.py"
phase integrity "$PYTHON" "$SCRIPT_DIR/integrity_after.py"
phase summary "$PYTHON" "$SCRIPT_DIR/summarize.py"

