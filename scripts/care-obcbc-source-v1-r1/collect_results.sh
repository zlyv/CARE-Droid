#!/usr/bin/env bash
set -euo pipefail
PROJECT="${OBCBC_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUT="${OBCBC_OUTPUT:-$PROJECT/outputs/care-obcbc-source-v1}"
for name in final_decision.json feasibility_decision.json ownership_audit.json context_coverage.json baseline_integrity_check.json; do
  echo "===== $name ====="
  if [[ -f "$OUT/$name" ]]; then cat "$OUT/$name"; else echo MISSING; fi
done
echo "===== report.md ====="
if [[ -f "$OUT/report.md" ]]; then cat "$OUT/report.md"; else echo MISSING; fi
