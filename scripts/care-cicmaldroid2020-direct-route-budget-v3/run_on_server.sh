#!/usr/bin/env bash
set -euo pipefail

PROJECT="${CIC_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${CIC_OUTPUT:-$PROJECT/outputs/care-cicmaldroid2020-direct-route-budget-v3}"
PYTHON="${CIC_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
export CUDA_VISIBLE_DEVICES="${CIC_GPU:-1}"
mkdir -p "$OUTPUT/logs" "$OUTPUT/checkpoints"

phase() {
  local name="$1"; shift
  if [[ -f "$OUTPUT/checkpoints/$name.done" ]]; then
    echo "CIC_PHASE_RESUME name=$name"
    return
  fi
  echo "CIC_PHASE_START name=$name epoch=$(date +%s)"
  "$@"
  touch "$OUTPUT/checkpoints/$name.done"
  echo "CIC_PHASE_DONE name=$name epoch=$(date +%s)"
}

cd "$PROJECT"
echo "CARE_CICMALDROID2020_DIRECT_ROUTE_BUDGET_PIPELINE_START output=$OUTPUT python=$PYTHON visible_gpu=$CUDA_VISIBLE_DEVICES"
phase syntax "$PYTHON" -m compileall -q "$SCRIPT_DIR"
phase selftest "$PYTHON" "$SCRIPT_DIR/selftest.py"
phase preflight "$PYTHON" "$SCRIPT_DIR/preflight.py"
phase archive_inventory "$PYTHON" "$SCRIPT_DIR/archive_inventory.py"
phase extract_apks "$PYTHON" "$SCRIPT_DIR/extract_apks.py"
phase manifest "$PYTHON" "$SCRIPT_DIR/build_manifest.py"
phase p31 "$PYTHON" "$SCRIPT_DIR/extract_p31.py"
phase stage1 "$PYTHON" "$SCRIPT_DIR/run_stage1.py"
phase stage2_builder_audit "$PYTHON" "$SCRIPT_DIR/locate_stage2_builders.py"
phase stage2_assets "$PYTHON" "$SCRIPT_DIR/extract_stage2_assets.py"
phase stage2 "$PYTHON" "$SCRIPT_DIR/run_stage2.py"
phase final_evaluation "$PYTHON" "$SCRIPT_DIR/evaluate.py"
phase integrity "$PYTHON" "$SCRIPT_DIR/integrity.py"
echo "CARE_CICMALDROID2020_DIRECT_ROUTE_BUDGET_PIPELINE_DONE"
echo "results=$OUTPUT/budget_results.csv"
echo "decision=$OUTPUT/final_decision.json"

