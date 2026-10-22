#!/usr/bin/env bash
set -euo pipefail
PROJECT="${ERA_PROJECT:-/home/boyuanjie/projects/tasa-droid}"; OUTPUT="${ERA_OUTPUT:-$PROJECT/outputs/care-svm-direct-c43-error-recovery-audit-v1}"
for file in final_decision.json annual_error_recovery.csv expert_ceiling.csv error_reason_by_year.csv error_reason_by_class_year.csv baseline_integrity_check.json REPORT.md; do echo "===== $file ====="; cat "$OUTPUT/$file"; done
