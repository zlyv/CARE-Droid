#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
O=outputs/care-p31-classifier-family-stage2-v1
echo "===== final_decision.json ====="
cat "$O/final_decision.json"
echo "===== 2017 full systems ====="
awk -F, 'NR==1 || $1=="source_2017_fixed_oof"' "$O/full_system_comparison.csv" | column -s, -t
echo "===== frozen future full systems ====="
awk -F, 'NR==1 || $1=="frozen_2017_to_future"' "$O/full_system_comparison.csv" | column -s, -t
echo "===== same-year full systems ====="
awk -F, 'NR==1 || $1=="independent_same_year_fixed_5fold"' "$O/full_system_comparison.csv" | column -s, -t
echo "===== integrity ====="
cat "$O/baseline_integrity_check.json"
echo "===== report ====="
cat "$O/REPORT.md"
