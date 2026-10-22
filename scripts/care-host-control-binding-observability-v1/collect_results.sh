#!/usr/bin/env bash
set -euo pipefail
PROJECT="${HCB_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="$PROJECT/outputs/care-host-control-binding-observability-v1"
for name in final_decision.json binding_coverage.json fold_binding_coverage.csv baseline_integrity_check.json phase_timing.json REPORT.md; do
  echo "===== $name ====="
  if [ -f "$OUTPUT/$name" ]; then cat "$OUTPUT/$name"; else echo "NOT_READY: $OUTPUT/$name"; fi
done
