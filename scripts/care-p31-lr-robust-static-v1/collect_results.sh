#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
O=outputs/care-p31-lr-robust-static-v1
echo "===== final_decision.json ====="
cat "$O/final_decision.json"
echo "===== source Stage-I representations ====="
column -s, -t "$O/source_stage1_metrics.csv"
echo "===== 2017 full systems ====="
awk -F, 'NR==1 || $1=="source_2017_fixed_oof"' "$O/full_system_comparison.csv" | column -s, -t
echo "===== frozen future full systems ====="
awk -F, 'NR==1 || $1=="frozen_2017_to_future"' "$O/full_system_comparison.csv" | column -s, -t
echo "===== same-year full systems ====="
awk -F, 'NR==1 || $1=="independent_same_year_fixed_5fold"' "$O/full_system_comparison.csv" | column -s, -t
echo "===== integrity ====="
cat "$O/baseline_integrity_check.json"
echo "===== probability drift (2021 malware rows) ====="
awk -F, 'NR==1 || ($2==2021 && $3==1)' "$O/probability_drift_by_year_label.csv" | column -s, -t
echo "===== report ====="
cat "$O/REPORT.md"
