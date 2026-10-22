#!/usr/bin/env bash
set -euo pipefail
PROJECT="${CARE_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${CARE_LINEAGE_OUTPUT:-${PROJECT}/outputs/care-p31-lineage-host-audit-v1}"
for file in final_decision.json lineage_summary.json p31_exposure_summary.json host_annotation_feasibility.json baseline_integrity_check.json; do
  echo "===== $file ====="
  cat "$OUTPUT/$file"
done
echo "===== p31_exposure_performance.csv ====="
column -s, -t "$OUTPUT/p31_exposure_performance.csv" || cat "$OUTPUT/p31_exposure_performance.csv"
echo "===== report.md ====="
cat "$OUTPUT/report.md"
