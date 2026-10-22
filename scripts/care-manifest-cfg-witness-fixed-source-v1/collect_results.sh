#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/boyuanjie/projects/tasa-droid
OUT=$ROOT/outputs/care_manifest_cfg_witness_fixed_source_v1
for name in protocol.json replay_validation.json final_decision.json baseline_integrity_check.json REPORT.md; do
  echo "===== $name ====="
  cat "$OUT/$name"
done
echo "===== W1 FUTURE MACRO COMPARISONS ====="
awk -F, 'NR==1 || $1=="FUTURE_MACRO_2018_2021"' "$OUT/w1_comparisons.csv"
