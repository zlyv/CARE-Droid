#!/usr/bin/env bash
set -euo pipefail
PROJECT="${RCTD_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="$PROJECT/outputs/care-direct-c43-route-conditioned-diagnostics-v1"
for name in final_decision.json frozen_B0_replay.json frozen_future_metrics.csv within_year_metrics.csv within_year_fold_metrics.csv baseline_integrity_check.json phase_timing.json REPORT.md; do
  if [ -f "$OUTPUT/$name" ]; then printf '\n===== %s =====\n' "$name"; cat "$OUTPUT/$name"; fi
done
