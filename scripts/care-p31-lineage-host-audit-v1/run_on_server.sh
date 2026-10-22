#!/usr/bin/env bash
set -euo pipefail

PROJECT="${CARE_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${CARE_LINEAGE_OUTPUT:-${PROJECT}/outputs/care-p31-lineage-host-audit-v1}"

if [[ ! "$OUTPUT" == "$PROJECT"/outputs/care-p31-lineage-host-audit-v* ]]; then
  echo "BLOCKED_BY_CONTRACT: output must be a new lineage audit directory under $PROJECT/outputs" >&2
  exit 1
fi

PYTHON_BIN="${CARE_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || true)"
fi
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "BLOCKED_BY_CONTRACT: Python interpreter not found" >&2
  exit 1
fi

mkdir -p "$OUTPUT/logs" "$OUTPUT/checkpoints"
export CARE_PROJECT="$PROJECT"
export CARE_LINEAGE_OUTPUT="$OUTPUT"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

run_phase() {
  local name="$1"
  shift
  local marker="$OUTPUT/checkpoints/${name}.done"
  if [[ -f "$marker" ]]; then
    echo "P31_LINEAGE_PHASE_RESUME name=$name"
    return 0
  fi
  echo "P31_LINEAGE_PHASE_START name=$name"
  "$@"
  touch "$marker"
  echo "P31_LINEAGE_PHASE_DONE name=$name"
}

echo "CARE_P31_LINEAGE_HOST_AUDIT_START python=$PYTHON_BIN output=$OUTPUT"
run_phase syntax "$PYTHON_BIN" -m py_compile "$SCRIPT_DIR/common.py" "$SCRIPT_DIR/fingerprints.py" "$SCRIPT_DIR/selftest.py" "$SCRIPT_DIR/preflight.py" "$SCRIPT_DIR/run_audit.py" "$SCRIPT_DIR/summarize.py"
run_phase selftest "$PYTHON_BIN" "$SCRIPT_DIR/selftest.py"
run_phase preflight "$PYTHON_BIN" "$SCRIPT_DIR/preflight.py"
run_phase audit "$PYTHON_BIN" "$SCRIPT_DIR/run_audit.py"
"$PYTHON_BIN" "$SCRIPT_DIR/summarize.py"
echo "CARE_P31_LINEAGE_HOST_AUDIT_DONE"
