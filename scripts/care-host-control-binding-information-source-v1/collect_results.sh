#!/usr/bin/env bash
set -euo pipefail
PROJECT="${HBI_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="$PROJECT/outputs/care-host-control-binding-information-source-v1"
for name in final_decision.json primary_source_metrics.csv primary_fold_metrics.csv paired_intervals.csv aggregate_replay_validation.json feature_validation.json baseline_integrity_check.json phase_timing.json REPORT.md; do
  echo "===== $name ====="
  if [ -f "$OUTPUT/$name" ]; then cat "$OUTPUT/$name"; else echo "NOT_READY: $OUTPUT/$name"; fi
done
