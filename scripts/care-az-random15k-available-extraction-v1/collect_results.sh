#!/usr/bin/env bash
set -euo pipefail
OUT=/home/boyuanjie/projects/tasa-droid/outputs/care_az_random15k_available_extraction_v1
for f in coverage_before.json new_stage2_asset_audit.json baseline_integrity_check.json final_decision.json; do
  echo "===== $f ====="; cat "$OUT/$f"
done
