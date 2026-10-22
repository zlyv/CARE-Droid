#!/usr/bin/env bash
set -euo pipefail
PROJECT="${PLRF_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
O="$PROJECT/outputs/care-p31-lr-source-safe-fusion-v1"
echo "===== final_decision.json ====="; cat "$O/final_decision.json"
echo "===== source_metrics.csv ====="; column -s, -t "$O/source_metrics.csv"
echo "===== paired_vs_raw.csv ====="; column -s, -t "$O/paired_vs_raw.csv"
echo "===== source_fold_metrics.csv ====="; column -s, -t "$O/source_fold_metrics.csv"
echo "===== integrity ====="; cat "$O/baseline_integrity_check.json"

