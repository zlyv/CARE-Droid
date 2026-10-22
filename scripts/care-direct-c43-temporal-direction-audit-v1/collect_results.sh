#!/usr/bin/env bash
set -euo pipefail
PROJECT="${TDA_PROJECT:-/home/boyuanjie/projects/tasa-droid}"; OUTPUT="${TDA_OUTPUT:-$PROJECT/outputs/care-direct-c43-temporal-direction-audit-v1}"
for file in final_decision.json conditional_direction_by_year.csv routed_oracle_threshold_diagnostic.csv expert_failure_confidence.csv baseline_integrity_check.json REPORT.md; do echo "===== $file ====="; cat "$OUTPUT/$file"; done
