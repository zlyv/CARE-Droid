#!/usr/bin/env bash
set -euo pipefail
PROJECT="${CRC_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${CRC_OUTPUT:-$PROJECT/outputs/care-svm-c43-representation-collision-audit-v1}"
for f in final_decision.json collision_audit.json replay_validation.json baseline_integrity_check.json REPORT.md; do
  echo "===== $f ====="; cat "$OUTPUT/$f"
done
echo "===== probe_pooled_results.csv ====="; column -s, -t "$OUTPUT/probe_pooled_results.csv" || cat "$OUTPUT/probe_pooled_results.csv"
echo "===== probe_direct_decision_diagnostics.csv ====="; column -s, -t "$OUTPUT/probe_direct_decision_diagnostics.csv" || cat "$OUTPUT/probe_direct_decision_diagnostics.csv"
