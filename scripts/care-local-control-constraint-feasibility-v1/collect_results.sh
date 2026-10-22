#!/usr/bin/env bash
set -euo pipefail
PROJECT="${LCC_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${LCC_OUTPUT:-$PROJECT/outputs/care-local-control-constraint-feasibility-v1}"
for name in final_decision.json coverage.json fold_coverage.csv baseline_integrity_check.json REPORT.md; do
  echo "===== $name ====="
  if [ -f "$OUTPUT/$name" ]; then cat "$OUTPUT/$name"; else echo "Not yet produced: $OUTPUT/$name"; fi
done
