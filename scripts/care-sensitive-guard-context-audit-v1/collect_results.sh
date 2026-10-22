#!/usr/bin/env bash
set -euo pipefail
PROJECT="${SGCA_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${SGCA_OUTPUT:-$PROJECT/outputs/care-sensitive-guard-context-audit-v1}"
for name in final_decision.json pilot_coverage.json baseline_integrity_check.json REPORT.md; do
  echo "===== $name ====="
  cat "$OUTPUT/$name"
done
