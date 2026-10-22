#!/usr/bin/env bash
set -euo pipefail
PROJECT="${PCF_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${PCF_OUTPUT:-$PROJECT/outputs/care-p31-classifier-family-v1}"
for name in final_decision.json source_metrics.csv source_paired_vs_rf.csv frozen_future_metrics.csv within_year_metrics.csv baseline_integrity_check.json report.md; do
  echo "===== $name ====="; if [[ -f "$OUTPUT/$name" ]]; then cat "$OUTPUT/$name"; else echo MISSING; fi
done

