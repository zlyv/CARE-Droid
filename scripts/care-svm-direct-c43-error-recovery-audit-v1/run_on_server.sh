#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${ERA_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${ERA_OUTPUT:-$PROJECT/outputs/care-svm-direct-c43-error-recovery-audit-v1}"
PYTHON="${ERA_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3 || true)"
[[ -n "$PYTHON" && -x "$PYTHON" ]] || { echo "BLOCKED_BY_CONTRACT: python unavailable"; exit 1; }
case "$(realpath -m "$OUTPUT")" in "$(realpath "$PROJECT")"/outputs/care-svm-direct-c43-error-recovery-audit-v*) ;; *) echo "BLOCKED_BY_CONTRACT: unsafe output=$OUTPUT"; exit 1 ;; esac
mkdir -p "$OUTPUT/logs"; export ERA_PROJECT="$PROJECT" ERA_OUTPUT="$OUTPUT" ERA_PYTHON="$PYTHON"
phase(){ local name="$1"; shift; echo "ERA_PHASE_START name=$name epoch=$(date +%s)"; "$@"; echo "ERA_PHASE_DONE name=$name epoch=$(date +%s)"; }
echo "CARE_SVM_DIRECT_C43_ERROR_RECOVERY_START output=$OUTPUT"
phase syntax "$PYTHON" -m py_compile "$SCRIPT_DIR"/*.py
phase selftest "$PYTHON" "$SCRIPT_DIR/selftest.py"
phase audit "$PYTHON" "$SCRIPT_DIR/audit.py"
echo "CARE_SVM_DIRECT_C43_ERROR_RECOVERY_DONE"
