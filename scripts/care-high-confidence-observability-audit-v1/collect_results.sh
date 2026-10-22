#!/usr/bin/env bash
set -euo pipefail
PROJECT="${HCOA_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${HCOA_OUTPUT:-$PROJECT/outputs/care-high-confidence-observability-audit-v1}"
for name in final_decision.json selection_audit.json paired_binary_results.csv paired_continuous_results.csv baseline_integrity_check.json REPORT.md; do
  echo "===== $name ====="; cat "$OUTPUT/$name"
done
