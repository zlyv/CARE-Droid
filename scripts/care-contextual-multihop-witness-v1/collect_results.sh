#!/usr/bin/env bash
set -euo pipefail
PROJECT="${CMW_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUT="${CMW_OUTPUT:-$PROJECT/outputs/care-contextual-multihop-witness-v1}"
for name in final_decision.json coverage_gate.json source_decision.json direction_probe_results.json source_end_to_end_results.json diagnostic_summary.json baseline_integrity_check.json report.md; do
  if [[ -f "$OUT/$name" ]]; then echo "===== $name ====="; cat "$OUT/$name"; fi
done
