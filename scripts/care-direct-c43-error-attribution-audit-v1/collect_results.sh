#!/usr/bin/env bash
set -euo pipefail
PROJECT="${ECA_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${ECA_OUTPUT:-$PROJECT/outputs/care-direct-c43-error-attribution-audit-v1}"
for file in final_decision.json baseline_integrity_check.json REPORT.md source_2017_group_summary.csv; do
  echo "===== $file ====="
  cat "$OUTPUT/$file"
done
echo "===== OUTPUT INVENTORY ====="
find "$OUTPUT" -maxdepth 1 -type f -printf '%f\n' | sort
