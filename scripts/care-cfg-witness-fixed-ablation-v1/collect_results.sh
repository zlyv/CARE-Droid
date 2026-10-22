#!/usr/bin/env bash
set -euo pipefail
OUT=/home/boyuanjie/projects/tasa-droid/outputs/care_cfg_witness_fixed_ablation_v1
for file in protocol.json smoke_test.json final_decision.json baseline_integrity_check.json REPORT.md; do
  echo "===== $file ====="
  cat "$OUT/$file"
done
