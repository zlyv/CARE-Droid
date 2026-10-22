#!/usr/bin/env bash
set -euo pipefail
PROJECT="${SICA_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${SICA_OUTPUT:-$PROJECT/outputs/care-static-identifiability-ceiling-audit-v1}"
for name in final_decision.json annual_error_funnel.csv confusion_error_funnel.csv source_collision_strata.csv source_collision_summary.json baseline_integrity_check.json REPORT.md; do
  echo "===== $name ====="; cat "$OUTPUT/$name"
done
