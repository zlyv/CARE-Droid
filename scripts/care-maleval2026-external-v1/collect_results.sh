#!/usr/bin/env bash
set -euo pipefail
O="${MALEVAL_OUTPUT:-/home/boyuanjie/projects/tasa-droid/outputs/care-maleval2026-external-v1}"
for f in dataset_audit.json stage1_fit_audit.json stage2_fit_audit.json final_decision.json baseline_integrity_check.json; do
  echo "===== $f ====="; [[ -f "$O/$f" ]] && cat "$O/$f" || echo MISSING
done
echo "===== external_metrics.csv ====="
[[ -f "$O/external_metrics.csv" ]] && column -s, -t "$O/external_metrics.csv" || echo MISSING
echo "===== REPORT.md ====="
[[ -f "$O/REPORT.md" ]] && cat "$O/REPORT.md" || echo MISSING

