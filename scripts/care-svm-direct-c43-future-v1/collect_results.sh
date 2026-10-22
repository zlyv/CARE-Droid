#!/usr/bin/env bash
set -euo pipefail
PROJECT="${SDC43F_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${SDC43F_OUTPUT:-$PROJECT/outputs/care-svm-direct-c43-future-v1}"
for file in final_decision.json future_metrics.csv direct_vs_bounded.csv fit_audit.json protocol.json baseline_integrity_check.json REPORT.md; do
  echo "===== $file ====="; cat "$OUTPUT/$file"
done
