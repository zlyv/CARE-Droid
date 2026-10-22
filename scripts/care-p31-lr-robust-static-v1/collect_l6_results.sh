#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
O=outputs/care-p31-lr-robust-static-l6-v1
echo "===== final_decision.json ====="
cat "$O/final_decision.json"
echo "===== L0 vs L6 all scopes ====="
column -s, -t "$O/full_system_comparison.csv"
echo "===== RF strong reference ====="
column -s, -t "$O/rf_strong_reference.csv"
echo "===== integrity ====="
cat "$O/baseline_integrity_check.json"
echo "===== report ====="
cat "$O/REPORT.md"
