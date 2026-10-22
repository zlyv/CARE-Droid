#!/usr/bin/env bash
set -euo pipefail
PROJECT="${SCS_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUT="${SCS_OUTPUT:-$PROJECT/outputs/care-sensitive-consequence-slice-v1}"
for name in final_decision.json coverage_gate.json direction_probe.json baseline_integrity_check.json report.md; do
  echo "===== $name ====="
  if [[ -f "$OUT/$name" ]]; then cat "$OUT/$name"; else echo "MISSING"; fi
done
