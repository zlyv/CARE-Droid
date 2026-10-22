#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${ECA_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${ECA_OUTPUT:-$PROJECT/outputs/care-direct-c43-error-attribution-audit-v1}"
PYTHON="${ECA_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3 || true)"
fi
[[ -n "$PYTHON" && -x "$PYTHON" ]] || { echo "BLOCKED_BY_CONTRACT: Python missing"; exit 1; }
case "$(realpath -m "$OUTPUT")" in
  "$(realpath "$PROJECT")"/outputs/care-direct-c43-error-attribution-audit-v*) ;;
  *) echo "BLOCKED_BY_CONTRACT: unsafe output directory $OUTPUT"; exit 1 ;;
esac
if [[ -e "$OUTPUT/final_decision.json" ]]; then
  echo "BLOCKED_BY_CONTRACT: completed audit exists, not overwriting"
  exit 1
fi
mkdir -p "$OUTPUT/logs"
export ECA_PROJECT="$PROJECT" ECA_OUTPUT="$OUTPUT"
phase() {
  local name="$1"
  shift
  echo "ECA_PHASE_START name=$name epoch=$(date +%s)"
  "$@"
  echo "ECA_PHASE_DONE name=$name epoch=$(date +%s)"
}
echo "CARE_DIRECT_C43_ERROR_ATTRIBUTION_START output=$OUTPUT"
phase syntax "$PYTHON" -m py_compile "$SCRIPT_DIR/run_audit.py" "$SCRIPT_DIR/validate_reviews.py" "$SCRIPT_DIR/selftest.py"
phase selftest "$PYTHON" "$SCRIPT_DIR/selftest.py"
phase source_only_audit "$PYTHON" "$SCRIPT_DIR/run_audit.py"
echo "CARE_DIRECT_C43_ERROR_ATTRIBUTION_DONE"
