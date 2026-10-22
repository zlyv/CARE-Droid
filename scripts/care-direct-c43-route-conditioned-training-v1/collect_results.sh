#!/usr/bin/env bash
set -euo pipefail
PROJECT="${RCT_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="$PROJECT/outputs/care-direct-c43-route-conditioned-training-v1"
for name in final_decision.json baseline_replay_validation.json source_results.csv source_fold_results.csv paired_comparisons.csv baseline_integrity_check.json phase_timing.json REPORT.md; do
  if [ -f "$OUTPUT/$name" ]; then
    printf '\n===== %s =====\n' "$name"
    cat "$OUTPUT/$name"
  fi
done
