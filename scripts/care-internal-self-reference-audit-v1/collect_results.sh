#!/usr/bin/env bash
set -euo pipefail
OUT=/home/boyuanjie/projects/tasa-droid/outputs/care_internal_self_reference_audit_v1
for file in final_decision.json frozen_model_intervention_results.csv error_association.csv baseline_integrity_check.json REPORT.md; do
  echo "===== $file ====="
  cat "$OUT/$file"
done
