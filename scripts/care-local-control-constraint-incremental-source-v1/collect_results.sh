#!/usr/bin/env bash
set -euo pipefail
PROJECT="${LCIP_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${LCIP_OUTPUT:-$PROJECT/outputs/care-local-control-constraint-incremental-source-v1}"
echo "Primary rows follow; ALL null repeats remain in source_metrics.csv and fold_metrics.csv."
for name in final_decision.json primary_source_metrics.csv primary_fold_metrics.csv paired_intervals.csv probe_training_audit.json feature_timing.json baseline_integrity_check.json REPORT.md; do
  echo "===== $name ====="
  if [ -f "$OUTPUT/$name" ]; then cat "$OUTPUT/$name"; else echo "Not yet produced: $OUTPUT/$name"; fi
done
