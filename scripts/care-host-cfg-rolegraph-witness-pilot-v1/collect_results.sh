#!/usr/bin/env bash
set -euo pipefail
OUT=/home/boyuanjie/projects/tasa-droid/artifacts/care_cfg_witness_pilot_2017
for file in project_inspection.md coverage_report.json redundancy_report.json final_decision.json baseline_integrity_check.json summary.md; do
  echo "===== $file ====="
  cat "$OUT/$file"
done
