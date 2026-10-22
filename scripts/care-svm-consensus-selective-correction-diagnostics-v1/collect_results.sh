#!/usr/bin/env bash
set -euo pipefail
OUT="${SCSCD_OUTPUT:-/home/boyuanjie/projects/tasa-droid/outputs/care-svm-consensus-selective-correction-diagnostics-v1}"
for f in diagnostic_decision.json matched_random_controls.json baseline_integrity_check.json report.md;do echo "===== $f =====";cat "$OUT/$f";done
echo "===== diagnostic_comparison.csv =====";column -s, -t "$OUT/diagnostic_comparison.csv" 2>/dev/null||cat "$OUT/diagnostic_comparison.csv"
echo "===== diagnostic_results.csv =====";column -s, -t "$OUT/diagnostic_results.csv" 2>/dev/null||cat "$OUT/diagnostic_results.csv"
