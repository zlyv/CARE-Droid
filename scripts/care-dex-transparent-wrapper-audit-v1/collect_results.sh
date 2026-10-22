#!/usr/bin/env bash
set -euo pipefail
PROJECT="${TW_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${TW_OUTPUT:-$PROJECT/outputs/care-dex-transparent-wrapper-audit-v1}"

for name in final_decision.json pilot_coverage.json baseline_integrity_check.json REPORT.md; do
  echo "===== $name ====="
  if [[ -f "$OUTPUT/$name" ]]; then
    cat "$OUTPUT/$name"
  else
    echo "MISSING: $OUTPUT/$name"
  fi
done

