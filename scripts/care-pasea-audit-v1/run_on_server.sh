#!/usr/bin/env bash
set -uo pipefail

PROJECT="${PASEA_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${PASEA_OUTPUT:-$PROJECT/outputs/care-pasea-audit-v1}"
mkdir -p "$OUTPUT/logs" "$OUTPUT/checkpoints"

if [[ -x /home/boyuanjie/envs/tif/bin/python ]]; then
  PY=/home/boyuanjie/envs/tif/bin/python
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  echo "BLOCKED_BY_CONTRACT: python3 not found" >&2
  exit 1
fi

export PASEA_PROJECT="$PROJECT"
export PASEA_OUTPUT="$OUTPUT"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

run_phase() {
  local name="$1"; shift
  local done="$OUTPUT/checkpoints/$name.done"
  if [[ -f "$done" ]]; then
    echo "PASEA_PHASE_RESUME name=$name"
    return 0
  fi
  echo "PASEA_PHASE_START name=$name epoch=$(date +%s)"
  if "$@"; then
    date -Is > "$done"
    echo "PASEA_PHASE_DONE name=$name epoch=$(date +%s)"
  else
    echo "PASEA_PHASE_FAIL name=$name" >&2
    return 1
  fi
}

echo "CARE_PASEA_AUDIT_START project=$PROJECT output=$OUTPUT python=$PY"
run_phase syntax "$PY" -m py_compile "$SCRIPT_DIR"/*.py || exit 1
run_phase selftest "$PY" "$SCRIPT_DIR/selftest.py" || exit 1
run_phase preflight "$PY" "$SCRIPT_DIR/preflight.py" || exit 1
run_phase manifest_before "$PY" "$SCRIPT_DIR/asset_manifest.py" before || exit 1
run_phase history "$PY" "$SCRIPT_DIR/history_audit.py" || exit 1
run_phase inventory "$PY" "$SCRIPT_DIR/inventory.py" || exit 1
run_phase summary "$PY" "$SCRIPT_DIR/summarize.py" || exit 1
run_phase manifest_after "$PY" "$SCRIPT_DIR/asset_manifest.py" after || exit 1

echo "CARE_PASEA_AUDIT_DONE"
echo "report=$OUTPUT/REPORT.md"
echo "decision=$OUTPUT/final_decision.json"

