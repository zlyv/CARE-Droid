#!/usr/bin/env bash
set -euo pipefail
O="${PLRPD_OUTPUT:-/home/boyuanjie/projects/tasa-droid/outputs/care-p31-lr-pairwise-diagnostics-v1}"
echo "===== final_decision.json ====="; cat "$O/final_decision.json"
echo "===== frozen_metrics.csv ====="; column -s, -t "$O/frozen_metrics.csv"
echo "===== within_metrics.csv ====="; column -s, -t "$O/within_metrics.csv"
echo "===== within_fold_metrics.csv ====="; column -s, -t "$O/within_fold_metrics.csv"

