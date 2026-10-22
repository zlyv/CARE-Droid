#!/usr/bin/env bash
set -euo pipefail
PROJECT="${FPR_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUT="${FPR_OUTPUT:-$PROJECT/outputs/care-fp-count-matched-residual-v1}"
for name in final_decision.json count_direction_audit.csv source_fold_results.csv baseline_integrity_check.json report.md; do
  echo "===== $name ====="
  if [[ -f "$OUT/$name" ]]; then cat "$OUT/$name"; else echo "MISSING"; fi
done
