#!/usr/bin/env bash
set -euo pipefail
PROJECT="${CSEF_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${CSEF_OUTPUT:-$PROJECT/outputs/care-c43-static-evidence-fidelity-audit-v1}"
for name in final_decision.json mechanism_summary.csv baseline_integrity_check.json REPORT.md; do
  echo "===== $name ====="
  cat "$OUTPUT/$name"
done
