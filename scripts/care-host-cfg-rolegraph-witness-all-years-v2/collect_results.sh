#!/usr/bin/env bash
set -euo pipefail
OUT=/home/boyuanjie/projects/tasa-droid/artifacts/care_cfg_witness_all_years_v2
for file in preflight.json coverage_by_year_scope.csv witness_direction_auc.csv final_decision.json baseline_integrity_check.json REPORT.md; do
  echo "===== $file ====="
  cat "$OUT/$file"
done
