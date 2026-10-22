#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python
S=scripts/care-droid-formal-v1
A=artifacts/care-droid-formal-v1
mkdir -p "$A"
export PYTHONWARNINGS=ignore
export LOGURU_LEVEL=ERROR
"$PY" "$S/selftest.py"
"$PY" "$S/audit_existing.py"
"$PY" "$S/build_protocol.py"
"$PY" "$S/reproduce.py"
if [[ ! -s "$A/formal_ablation_results.csv" || ! -s "$A/routing_analysis.csv" || ! -s "$A/budget_curve.csv" ]]; then "$PY" "$S/run_ablations.py"; else echo FORMAL_ABLATIONS_RESUME_EXISTING; fi
if [[ ! -s "$A/cost_summary.csv" || ! -s "$A/environment_manifest.json" ]]; then "$PY" "$S/cost_audit.py"; else echo COST_ANALYSIS_RESUME_EXISTING; fi
if [[ ! -s "$A/internal_ablation_results.csv" ]]; then "$PY" "$S/internal_ablation_audit.py"; else echo INTERNAL_ABLATION_AUDIT_RESUME_EXISTING; fi
"$PY" "$S/summarize.py"
"$PY" "$S/freeze.py"
echo CARE_DROID_FORMAL_V1_PIPELINE_DONE
echo artifacts=$A
echo report=$A/REPORT.txt
echo final_summary=$A/final_summary.json
echo freeze_manifest=$A/formal_freeze_manifest.json

