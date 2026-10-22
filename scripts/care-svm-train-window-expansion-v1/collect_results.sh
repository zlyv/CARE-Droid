#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
O="outputs/care-svm-train-window-expansion-v1"
for file in final_decision.json evaluation_decision.json baseline_integrity_check.json REPORT.md; do
  echo "===== $file ====="
  cat "$O/$file"
done
echo "===== train_window_results.csv ====="
column -s, -t "$O/train_window_results.csv"
