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
phase syntax python -m py_compile "$SCRIPT_DIR"/*.py
phase selftest python "$SCRIPT_DIR/selftest.py"
phase asset_audit python "$SCRIPT_DIR/audit_assets.py"
phase baseline_reproduction python "$SCRIPT_DIR/reproduce_baseline.py"
phase contracts_5 python "$SCRIPT_DIR/extract_contracts.py" --limit 5
phase contracts_5_audit python "$SCRIPT_DIR/validate_contracts.py" --expected 5
phase contracts_25 python "$SCRIPT_DIR/extract_contracts.py" --limit 25
phase contracts_25_audit python "$SCRIPT_DIR/validate_contracts.py" --expected 25
phase contracts_799 python "$SCRIPT_DIR/extract_contracts.py"
phase component_scope python "$SCRIPT_DIR/build_component_scope.py"
phase formal_attention python "$SCRIPT_DIR/replay_attention.py"
phase preflight python "$SCRIPT_DIR/preflight.py"
phase source_five_fold python "$SCRIPT_DIR/run_source.py"
phase timing python "$SCRIPT_DIR/summarize_timing.py"
phase integrity python "$SCRIPT_DIR/integrity_after.py"
phase source_summary python "$SCRIPT_DIR/summarize.py"

echo "CARE_CCBC_SOURCE_PIPELINE_DONE"
echo "report=$CCBC_OUTPUT/report.md"
echo "final_summary=$CCBC_OUTPUT/final_summary.json"
echo "protocol_hash=$(tr -d '\r\n' < "$CCBC_OUTPUT/protocol_hash.txt")"
