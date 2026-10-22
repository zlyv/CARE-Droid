#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/boyuanjie/projects/tasa-droid
OUT=$ROOT/outputs/care_cfg_witness_fixed_source_future_v2
for name in protocol.json replay_validation.json final_decision.json baseline_integrity_check.json REPORT.md; do
  echo "===== $name ====="
  cat "$OUT/$name"
done
