#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CCBC_PROJECT="${CCBC_PROJECT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export CCBC_OUTPUT="${CCBC_OUTPUT:-$CCBC_PROJECT/outputs/care-ccbc-source-v1}"

case "$(realpath -m "$CCBC_OUTPUT")" in
  "$(realpath -m "$CCBC_PROJECT")"/outputs/care-ccbc-source-v*) ;;
  *) echo "BLOCKED_BY_CONTRACT: unsafe CCBC_OUTPUT=$CCBC_OUTPUT"; exit 2 ;;
esac

mkdir -p "$CCBC_OUTPUT/cache/contracts" "$CCBC_OUTPUT/logs" "$CCBC_OUTPUT/checkpoints"
exec > >(tee -a "$CCBC_OUTPUT/logs/run.log") 2>&1

if [[ -n "${CCBC_PYTHON:-}" ]]; then
  PYTHON_BIN="$CCBC_PYTHON"
elif [[ -x /home/boyuanjie/envs/tif/bin/python ]]; then
  PYTHON_BIN=/home/boyuanjie/envs/tif/bin/python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "BLOCKED_BY_CONTRACT: no Python interpreter; set CCBC_PYTHON=/absolute/path/to/python"
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "BLOCKED_BY_CONTRACT: Python is not executable: $PYTHON_BIN"
  exit 2
fi
echo "CCBC_PYTHON=$PYTHON_BIN"

phase() {
  local name="$1"; shift
  local started finished status
  started="$(date +%s)"
  echo "CCBC_PHASE_START name=$name epoch=$started"
  if "$@"; then status=PASS; else status=FAIL; fi
  finished="$(date +%s)"
  printf '{"phase":"%s","started_epoch":%s,"finished_epoch":%s,"wall_seconds":%s,"status":"%s"}\n' \
    "$name" "$started" "$finished" "$((finished-started))" "$status" >> "$CCBC_OUTPUT/phase_timing.jsonl"
  if [[ "$status" != PASS ]]; then
    echo "CCBC_PHASE_FAIL name=$name"
    exit 1
  fi
  echo "CCBC_PHASE_DONE name=$name wall_seconds=$((finished-started))"
}

echo "CARE_CCBC_SOURCE_PIPELINE_START project=$CCBC_PROJECT output=$CCBC_OUTPUT"
phase syntax "$PYTHON_BIN" -m py_compile "$SCRIPT_DIR"/*.py
phase selftest "$PYTHON_BIN" "$SCRIPT_DIR/selftest.py"
phase asset_audit "$PYTHON_BIN" "$SCRIPT_DIR/audit_assets.py"
phase baseline_reproduction "$PYTHON_BIN" "$SCRIPT_DIR/reproduce_baseline.py"
phase contracts_5 "$PYTHON_BIN" "$SCRIPT_DIR/extract_contracts.py" --limit 5
phase contracts_5_audit "$PYTHON_BIN" "$SCRIPT_DIR/validate_contracts.py" --expected 5
phase contracts_25 "$PYTHON_BIN" "$SCRIPT_DIR/extract_contracts.py" --limit 25
phase contracts_25_audit "$PYTHON_BIN" "$SCRIPT_DIR/validate_contracts.py" --expected 25
phase contracts_799 "$PYTHON_BIN" "$SCRIPT_DIR/extract_contracts.py"
phase component_scope "$PYTHON_BIN" "$SCRIPT_DIR/build_component_scope.py"
phase formal_attention "$PYTHON_BIN" "$SCRIPT_DIR/replay_attention.py"
phase preflight "$PYTHON_BIN" "$SCRIPT_DIR/preflight.py"
phase source_five_fold "$PYTHON_BIN" "$SCRIPT_DIR/run_source.py"
phase timing "$PYTHON_BIN" "$SCRIPT_DIR/summarize_timing.py"
phase integrity "$PYTHON_BIN" "$SCRIPT_DIR/integrity_after.py"
phase source_summary "$PYTHON_BIN" "$SCRIPT_DIR/summarize.py"

echo "CARE_CCBC_SOURCE_PIPELINE_DONE"
echo "report=$CCBC_OUTPUT/report.md"
echo "final_summary=$CCBC_OUTPUT/final_summary.json"
echo "protocol_hash=$(tr -d '\r\n' < "$CCBC_OUTPUT/protocol_hash.txt")"
