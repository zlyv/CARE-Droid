#!/usr/bin/env bash
set -euo pipefail
PROJECT="${CELA_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUT="${CELA_OUTPUT:-$PROJECT/outputs/care-consensus-error-lineage-label-audit-v1}"
for file in final_decision.json baseline_integrity_check.json preflight.json; do
  echo "===== $file ====="
  cat "$OUT/$file"
done
echo "===== lineage_stratum_summary.csv ====="
column -s, -t "$OUT/lineage_stratum_summary.csv" 2>/dev/null || cat "$OUT/lineage_stratum_summary.csv"
echo "===== REPORT.md ====="
cat "$OUT/REPORT.md"
