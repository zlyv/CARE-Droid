#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${SGCA_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${SGCA_OUTPUT:-$PROJECT/outputs/care-sensitive-guard-context-audit-v1}"
PYTHON="${SGCA_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"

if [[ ! -x "$PYTHON" ]]; then PYTHON="$(command -v python3 || true)"; fi
if [[ -z "$PYTHON" || ! -x "$PYTHON" ]]; then
  echo "BLOCKED_BY_CONTRACT: no Python interpreter" >&2
  exit 1
fi
case "$(realpath -m "$OUTPUT")" in
  "$(realpath "$PROJECT")"/outputs/care-sensitive-guard-context-audit-v*) ;;
  *) echo "BLOCKED_BY_CONTRACT: unsafe output=$OUTPUT" >&2; exit 1 ;;
esac

mkdir -p "$OUTPUT/logs" "$OUTPUT/checkpoints"
export SGCA_PROJECT="$PROJECT" SGCA_OUTPUT="$OUTPUT" SGCA_PYTHON="$PYTHON"

run_phase() {
  local name="$1"; shift
  echo "SGCA_PHASE_START name=$name epoch=$(date +%s)"
  if "$@"; then
    echo "SGCA_PHASE_DONE name=$name epoch=$(date +%s)"
  else
    echo "SGCA_PHASE_FAIL name=$name epoch=$(date +%s)"
    return 1
  fi
}

echo "CARE_SENSITIVE_GUARD_CONTEXT_AUDIT_START output=$OUTPUT python=$PYTHON"
run_phase syntax "$PYTHON" -m py_compile "$SCRIPT_DIR"/*.py
run_phase selftest "$PYTHON" "$SCRIPT_DIR/selftest.py"
run_phase preflight "$PYTHON" "$SCRIPT_DIR/preflight.py"
run_phase pilot "$PYTHON" "$SCRIPT_DIR/audit.py"
echo "CARE_SENSITIVE_GUARD_CONTEXT_AUDIT_DONE"
