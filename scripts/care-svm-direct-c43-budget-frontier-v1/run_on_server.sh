#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${BRF_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${BRF_OUTPUT:-$PROJECT/outputs/care-svm-direct-c43-budget-frontier-v1}"
PYTHON="${BRF_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3 || true)"
[[ -n "$PYTHON" && -x "$PYTHON" ]] || { echo "BLOCKED_BY_CONTRACT: python unavailable"; exit 1; }
case "$(realpath -m "$OUTPUT")" in
  "$(realpath "$PROJECT")"/outputs/care-svm-direct-c43-budget-frontier-v*) ;;
  *) echo "BLOCKED_BY_CONTRACT: unsafe output=$OUTPUT"; exit 1 ;;
esac
mkdir -p "$OUTPUT/logs"
export BRF_PROJECT="$PROJECT" BRF_OUTPUT="$OUTPUT" BRF_PYTHON="$PYTHON"
phase(){ local name="$1"; shift; echo "BRF_PHASE_START name=$name epoch=$(date +%s)"; "$@"; echo "BRF_PHASE_DONE name=$name epoch=$(date +%s)"; }
echo "CARE_SVM_DIRECT_C43_BUDGET_FRONTIER_START output=$OUTPUT python=$PYTHON"
phase syntax "$PYTHON" -m py_compile "$SCRIPT_DIR"/*.py
phase selftest "$PYTHON" "$SCRIPT_DIR/selftest.py"
phase preflight "$PYTHON" "$SCRIPT_DIR/preflight.py"
phase frontier "$PYTHON" "$SCRIPT_DIR/run_frontier.py"
echo "CARE_SVM_DIRECT_C43_BUDGET_FRONTIER_DONE"

