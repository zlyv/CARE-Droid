#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${HCOA_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${HCOA_OUTPUT:-$PROJECT/outputs/care-high-confidence-observability-audit-v1}"
PYTHON="${HCOA_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
if [[ ! -x "$PYTHON" ]]; then PYTHON="$(command -v python3 || true)"; fi
if [[ -z "$PYTHON" || ! -x "$PYTHON" ]]; then echo "BLOCKED_BY_CONTRACT: no Python" >&2; exit 1; fi
case "$(realpath -m "$OUTPUT")" in
  "$(realpath "$PROJECT")"/outputs/care-high-confidence-observability-audit-v*) ;;
  *) echo "BLOCKED_BY_CONTRACT: unsafe output=$OUTPUT" >&2; exit 1 ;;
esac
mkdir -p "$OUTPUT/logs" "$OUTPUT/checkpoints"
export HCOA_PROJECT="$PROJECT" HCOA_OUTPUT="$OUTPUT" HCOA_PYTHON="$PYTHON"
run_phase() {
  local name="$1"; shift
  echo "HCOA_PHASE_START name=$name epoch=$(date +%s)"
  if "$@"; then echo "HCOA_PHASE_DONE name=$name epoch=$(date +%s)";
  else echo "HCOA_PHASE_FAIL name=$name epoch=$(date +%s)"; return 1; fi
}
echo "CARE_HIGH_CONFIDENCE_OBSERVABILITY_AUDIT_START output=$OUTPUT python=$PYTHON"
run_phase syntax "$PYTHON" -m py_compile "$SCRIPT_DIR"/*.py
run_phase selftest "$PYTHON" "$SCRIPT_DIR/selftest.py"
run_phase preflight "$PYTHON" "$SCRIPT_DIR/preflight.py"
run_phase audit "$PYTHON" "$SCRIPT_DIR/run_audit.py"
echo "CARE_HIGH_CONFIDENCE_OBSERVABILITY_AUDIT_DONE"
