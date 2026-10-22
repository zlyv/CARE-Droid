#!/usr/bin/env bash
set -euo pipefail
O="${CIC_OUTPUT:-/home/boyuanjie/projects/tasa-droid/outputs/care-cicmaldroid2020-direct-route-budget-v3}"
for f in archive_inventory.json extraction_summary.json dataset_audit.json p31_exclusion_audit.json final_decision.json stage1_fit_audit.json stage2_asset_audit.json stage2_fit_audit.json baseline_integrity_check.json; do
  echo "===== $f ====="; [[ -f "$O/$f" ]] && cat "$O/$f" || echo MISSING
done
echo "===== budget_results.csv ====="
[[ -f "$O/budget_results.csv" ]] && column -s, -t "$O/budget_results.csv" || echo MISSING
echo "===== cohort_budget_results.csv ====="
[[ -f "$O/cohort_budget_results.csv" ]] && column -s, -t "$O/cohort_budget_results.csv" || echo MISSING
echo "===== REPORT.md ====="
[[ -f "$O/REPORT.md" ]] && cat "$O/REPORT.md" || echo MISSING
