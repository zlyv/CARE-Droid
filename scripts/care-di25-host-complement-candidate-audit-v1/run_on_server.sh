#!/usr/bin/env bash
set -euo pipefail

cd /home/boyuanjie/projects/tasa-droid
ROOT="artifacts/care-di25-host-complement-candidate-audit-v1"
SCRIPT="scripts/care-di25-host-complement-candidate-audit-v1"
PY="/home/boyuanjie/envs/tif/bin/python"

mkdir -p "$ROOT"/{preflight,features,predictions,reports,audits}

"$PY" "$SCRIPT/selftest.py"
"$PY" "$SCRIPT/preflight.py"
"$PY" "$SCRIPT/extract_candidate_features.py"
"$PY" "$SCRIPT/run_candidate_probes.py"
"$PY" "$SCRIPT/analyze_conditional_rescue.py"
"$PY" "$SCRIPT/summarize_candidate_audit.py"

echo "HOST_COMPLEMENT_CANDIDATE_AUDIT_DONE"
echo "Report: $ROOT/reports/REPORT.txt"
