#!/usr/bin/env bash
set -euo pipefail
PROJECT="${PASEA_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${PASEA_OUTPUT:-$PROJECT/outputs/care-pasea-audit-v1}"
for name in final_decision.json existing_experiment_audit.json environment_inventory.json baseline_integrity_check.json REPORT.md; do
  echo "===== $name ====="
  if [[ -f "$OUTPUT/$name" ]]; then
    cat "$OUTPUT/$name"
  else
    echo "MISSING: $OUTPUT/$name"
  fi
done

