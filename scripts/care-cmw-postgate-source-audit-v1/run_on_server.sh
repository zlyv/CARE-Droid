#!/usr/bin/env bash
set -euo pipefail

PROJECT="${CMW_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${CMW_POSTGATE_OUTPUT:-$PROJECT/outputs/care-cmw-postgate-source-audit-v1}"
PY="${CMW_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
export CMW_PROJECT="$PROJECT" CMW_POSTGATE_OUTPUT="$OUT"
mkdir -p "$OUT/logs"
exec > >(tee -a "$OUT/logs/run.log") 2>&1

echo "CARE_CMW_POSTGATE_PIPELINE_START output=$OUT python=$PY"
"$PY" -m py_compile "$SCRIPT_DIR"/*.py
"$PY" "$SCRIPT_DIR/selftest.py"
"$PY" "$SCRIPT_DIR/preflight.py"
"$PY" "$SCRIPT_DIR/run_audit.py"
"$PY" "$SCRIPT_DIR/finalize.py"
"$PY" "$SCRIPT_DIR/summarize.py"
echo "CARE_CMW_POSTGATE_SOURCE_AUDIT_DONE"
echo "report=$OUT/report.md"
echo "decision=$OUT/final_decision.json"
