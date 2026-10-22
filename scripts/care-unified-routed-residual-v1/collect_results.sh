#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
O=outputs/care-unified-routed-residual-v1
echo "===== final_decision.json ====="
cat "$O/final_decision.json"
echo "===== routed_200_results.csv ====="
column -s, -t "$O/routed_200_results.csv" || cat "$O/routed_200_results.csv"
echo "===== direction_probe_folds.csv ====="
column -s, -t "$O/direction_probe_folds.csv" || cat "$O/direction_probe_folds.csv"
echo "===== baseline_integrity_check.json ====="
cat "$O/baseline_integrity_check.json"
