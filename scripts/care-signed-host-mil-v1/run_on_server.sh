#!/usr/bin/env bash
set -euo pipefail
PROJECT="${CSHM_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${CSHM_OUTPUT:-$PROJECT/outputs/care-signed-host-mil-v1}"
PYTHON="${CSHM_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
mkdir -p "$OUTPUT/logs" "$OUTPUT/checkpoints"

phase(){
  local name="$1"; shift
  if [[ -f "$OUTPUT/checkpoints/$name.done" ]]; then echo "CSHM_PHASE_RESUME name=$name"; return; fi
  echo "CSHM_PHASE_START name=$name epoch=$(date +%s)"
  "$@" 2>&1 | tee -a "$OUTPUT/logs/$name.log"
  touch "$OUTPUT/checkpoints/$name.done"
  echo "CSHM_PHASE_DONE name=$name epoch=$(date +%s)"
}

cd "$PROJECT"
echo "CARE_SIGNED_HOST_MIL_PIPELINE_START output=$OUTPUT python=$PYTHON cuda=${CUDA_VISIBLE_DEVICES:-default}"
phase syntax "$PYTHON" -m compileall -q "$SCRIPT_DIR"
phase selftest "$PYTHON" "$SCRIPT_DIR/selftest.py"
phase preflight "$PYTHON" "$SCRIPT_DIR/preflight.py"
phase source "$PYTHON" "$SCRIPT_DIR/run_source.py"
phase integrity "$PYTHON" "$SCRIPT_DIR/integrity_after.py"
phase summary "$PYTHON" "$SCRIPT_DIR/summarize.py"

