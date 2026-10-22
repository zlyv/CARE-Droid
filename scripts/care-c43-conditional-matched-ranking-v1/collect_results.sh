#!/usr/bin/env bash
set -euo pipefail
O="${CCMR_OUTPUT:-/home/boyuanjie/projects/tasa-droid/outputs/care-c43-conditional-matched-ranking-v1}"
for f in final_decision.json baseline_integrity_check.json report.md; do echo "===== $f ====="; cat "$O/$f"; done
echo "===== fold_results.csv ====="; column -s, -t "$O/fold_results.csv" || cat "$O/fold_results.csv"
