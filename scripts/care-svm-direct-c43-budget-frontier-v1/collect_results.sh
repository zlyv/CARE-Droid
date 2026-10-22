#!/usr/bin/env bash
set -euo pipefail
PROJECT="${BRF_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${BRF_OUTPUT:-$PROJECT/outputs/care-svm-direct-c43-budget-frontier-v1}"
for name in final_decision.json budget_frontier.csv marginal_budget_response.csv baseline_integrity_check.json REPORT.md; do
  echo "===== $name ====="
  [[ -f "$OUTPUT/$name" ]] && cat "$OUTPUT/$name" || echo "MISSING: $OUTPUT/$name"
done

