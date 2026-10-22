#!/usr/bin/env bash
set -euo pipefail

PROJECT="${MALEVAL_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${MALEVAL_OUTPUT:-$PROJECT/outputs/care-maleval2026-external-v1}"
PYTHON="${MALEVAL_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
mkdir -p "$OUTPUT/logs" "$OUTPUT/checkpoints"

phase() {
  local name="$1"; shift
  if [[ -f "$OUTPUT/checkpoints/$name.done" ]]; then
    echo "MALEVAL_PHASE_RESUME name=$name"
    return
  fi
  echo "MALEVAL_PHASE_START name=$name epoch=$(date +%s)"
  "$@"
  touch "$OUTPUT/checkpoints/$name.done"
  echo "MALEVAL_PHASE_DONE name=$name epoch=$(date +%s)"
}

cd "$PROJECT"
echo "CARE_MALEVAL2026_EXTERNAL_PIPELINE_START output=$OUTPUT python=$PYTHON"
phase syntax "$PYTHON" -m compileall -q "$SCRIPT_DIR"
phase selftest "$PYTHON" "$SCRIPT_DIR/selftest.py"
phase preflight "$PYTHON" "$SCRIPT_DIR/preflight.py"
phase manifest "$PYTHON" "$SCRIPT_DIR/build_manifest.py"
phase p31 "$PYTHON" "$SCRIPT_DIR/extract_p31.py"
phase stage1 "$PYTHON" "$SCRIPT_DIR/run_stage1.py"
phase stage1_evaluation "$PYTHON" "$SCRIPT_DIR/evaluate.py"
phase stage2_builder_audit "$PYTHON" "$SCRIPT_DIR/locate_stage2_builders.py"
phase stage2_assets "$PYTHON" "$SCRIPT_DIR/extract_stage2_assets.py"
phase stage2 "$PYTHON" "$SCRIPT_DIR/run_stage2.py"
phase final_evaluation "$PYTHON" "$SCRIPT_DIR/evaluate.py"
phase integrity "$PYTHON" "$SCRIPT_DIR/integrity.py"
echo "CARE_MALEVAL2026_EXTERNAL_PIPELINE_DONE"
echo "results=$OUTPUT/external_metrics.csv"
echo "decision=$OUTPUT/final_decision.json"

