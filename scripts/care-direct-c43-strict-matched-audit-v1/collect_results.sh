#!/usr/bin/env bash
set -euo pipefail
PROJECT="${SMCIA_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${SMCIA_OUTPUT:-$PROJECT/outputs/care-direct-c43-strict-matched-audit-v1}"
for f in final_decision.json strict_matched_findings.json replay_validation.json baseline_integrity_check.json REPORT.md; do
  echo "===== $f ====="; cat "$OUTPUT/$f"
done
echo "===== intervention_summary.csv ====="
column -s, -t "$OUTPUT/intervention_summary.csv" || cat "$OUTPUT/intervention_summary.csv"

