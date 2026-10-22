#!/usr/bin/env bash
set -euo pipefail
PROJECT="${CMW_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUT="${CMW_POSTGATE_OUTPUT:-$PROJECT/outputs/care-cmw-postgate-source-audit-v1}"
for name in final_decision.json direction_results.json input_integrity_check.json report.md; do
  if [[ -f "$OUT/$name" ]]; then echo "===== $name ====="; cat "$OUT/$name"; fi
done
