#!/usr/bin/env bash
set -euo pipefail
PROJECT="${CSCUA_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${CSCUA_OUTPUT:-$PROJECT/outputs/care-svm-c43-unfixed-audit-v1}"
for name in final_decision.json routed_error_category_summary.csv routed_error_fold_summary.csv descriptive_scale_sweep.csv baseline_integrity_check.json REPORT.md; do
  echo "===== $name ====="
  cat "$OUTPUT/$name"
done
