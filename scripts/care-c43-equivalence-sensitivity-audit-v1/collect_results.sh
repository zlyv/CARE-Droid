#!/usr/bin/env bash
set -u
PROJECT="${CESA_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${CESA_OUTPUT:-$PROJECT/outputs/care-c43-equivalence-sensitivity-audit-v2}"
for file in final_decision.json replay_validation.json transformed_s7_protocol.json transform_summary.csv fold_summary.csv baseline_integrity_check.json report.md; do
  echo "===== $file ====="
  if [[ -f "$OUTPUT/$file" ]]; then cat "$OUTPUT/$file"; else echo "MISSING: $OUTPUT/$file"; fi
done
