#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PROJECT="${TDA_PROJECT:-/home/boyuanjie/projects/tasa-droid}"; OUTPUT="${TDA_OUTPUT:-$PROJECT/outputs/care-direct-c43-temporal-direction-audit-v1}"; PYTHON="${TDA_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3 || true)"; [[ -n "$PYTHON" && -x "$PYTHON" ]] || { echo "BLOCKED_BY_CONTRACT: python unavailable"; exit 1; }
case "$(realpath -m "$OUTPUT")" in "$(realpath "$PROJECT")"/outputs/care-direct-c43-temporal-direction-audit-v*) ;; *) echo "BLOCKED_BY_CONTRACT: unsafe output=$OUTPUT"; exit 1 ;; esac
mkdir -p "$OUTPUT/logs"; export TDA_PROJECT="$PROJECT" TDA_OUTPUT="$OUTPUT" TDA_PYTHON="$PYTHON"
phase(){ local name="$1"; shift; echo "TDA_PHASE_START name=$name epoch=$(date +%s)"; "$@"; echo "TDA_PHASE_DONE name=$name epoch=$(date +%s)"; }
echo "CARE_DIRECT_C43_TEMPORAL_DIRECTION_START output=$OUTPUT"
phase syntax "$PYTHON" -m py_compile "$SCRIPT_DIR"/*.py
phase selftest "$PYTHON" "$SCRIPT_DIR/selftest.py"
phase audit "$PYTHON" "$SCRIPT_DIR/run_audit.py"
echo "CARE_DIRECT_C43_TEMPORAL_DIRECTION_DONE"
