#!/usr/bin/env bash
set -euo pipefail
PROJECT="${SBMA_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUT="${SBMA_OUTPUT:-$PROJECT/outputs/care-static-blindspot-modality-audit-v1}"
for file in final_decision.json feature_extraction.json baseline_integrity_check.json;do echo "===== $file =====";cat "$OUT/$file";done
echo "===== source_modality_results.csv ====="
column -s, -t "$OUT/source_modality_results.csv" 2>/dev/null || cat "$OUT/source_modality_results.csv"
echo "===== future_diagnostic_results.csv ====="
column -s, -t "$OUT/future_diagnostic_results.csv" 2>/dev/null || cat "$OUT/future_diagnostic_results.csv"
echo "===== REPORT.md =====";cat "$OUT/REPORT.md"
