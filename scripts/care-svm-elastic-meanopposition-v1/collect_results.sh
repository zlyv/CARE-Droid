#!/usr/bin/env bash
set -euo pipefail
PROJECT="${EMO_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${EMO_OUTPUT:-$PROJECT/outputs/care-svm-elastic-meanopposition-v1}"
for file in final_decision.json source_metrics.csv future_metrics.csv paired_comparisons.csv baseline_integrity_check.json REPORT.md; do
  echo "===== $file ====="
  cat "$OUTPUT/$file"
done
