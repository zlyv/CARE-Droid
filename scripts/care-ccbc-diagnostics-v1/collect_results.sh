#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-outputs/care-ccbc-diagnostics-v1}"
for file in REPORT.txt final_diagnostic_summary.json future_component_mapping_audit.json diagnostic_integrity_check.json; do
  echo "===== $file ====="
  if [[ -s "$OUT/$file" ]]; then cat "$OUT/$file"; else echo "MISSING: $OUT/$file"; fi
done
echo "===== FROZEN POOLED ====="
column -s, -t < "$OUT/frozen_metrics.csv" | grep -E 'pooled|arm' || true
echo "===== SAME-YEAR POOLED ====="
column -s, -t < "$OUT/within_metrics.csv" | grep -E 'pooled|arm' || true
