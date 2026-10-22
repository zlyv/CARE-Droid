#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${EMO_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${EMO_OUTPUT:-$PROJECT/outputs/care-svm-elastic-meanopposition-v1}"
PYTHON="${EMO_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3 || true)"
[[ -n "$PYTHON" && -x "$PYTHON" ]] || { echo "BLOCKED_BY_CONTRACT: python unavailable"; exit 1; }
case "$(realpath -m "$OUTPUT")" in
  "$(realpath "$PROJECT")"/outputs/care-svm-elastic-meanopposition-v*) ;;
  *) echo "BLOCKED_BY_CONTRACT: unsafe output=$OUTPUT"; exit 1 ;;
esac
mkdir -p "$OUTPUT/logs"
export EMO_PROJECT="$PROJECT" EMO_OUTPUT="$OUTPUT" EMO_PYTHON="$PYTHON"
phase(){ local name="$1"; shift; echo "EMO_PHASE_START name=$name epoch=$(date +%s)"; "$@"; echo "EMO_PHASE_DONE name=$name epoch=$(date +%s)"; }
echo "CARE_SVM_ELASTIC_MEANOPPOSITION_START output=$OUTPUT python=$PYTHON"
phase syntax "$PYTHON" -m py_compile "$SCRIPT_DIR"/*.py
phase selftest "$PYTHON" "$SCRIPT_DIR/selftest.py"
phase preflight "$PYTHON" "$SCRIPT_DIR/preflight.py"
phase freeze_policy "$PYTHON" "$SCRIPT_DIR/freeze_policy.py"
phase evaluate "$PYTHON" "$SCRIPT_DIR/evaluate.py"
echo "CARE_SVM_ELASTIC_MEANOPPOSITION_DONE"
