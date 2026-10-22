#!/usr/bin/env bash
set -euo pipefail
PROJECT="${SCSC_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUT="$PROJECT/outputs/care-svm-consensus-selective-correction-v1"
for file in source_decision.json matched_random_control.json baseline_integrity_check.json report.md; do
  echo "===== $file ====="
  cat "$OUT/$file"
done
echo "===== source_results.csv ====="
column -s, -t "$OUT/source_results.csv" 2>/dev/null || cat "$OUT/source_results.csv"
echo "===== source_fold_results.csv ====="
column -s, -t "$OUT/source_fold_results.csv" 2>/dev/null || cat "$OUT/source_fold_results.csv"
echo "===== trust_signal_audit.csv ====="
column -s, -t "$OUT/trust_signal_audit.csv" 2>/dev/null || cat "$OUT/trust_signal_audit.csv"
