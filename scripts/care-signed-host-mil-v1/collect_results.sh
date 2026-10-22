#!/usr/bin/env bash
set -euo pipefail
O="${CSHM_OUTPUT:-/home/boyuanjie/projects/tasa-droid/outputs/care-signed-host-mil-v1}"
for f in preflight.json direction_gate.json final_decision.json baseline_integrity_check.json; do
  echo "===== $f ====="; [[ -f "$O/$f" ]] && cat "$O/$f" || echo MISSING
done
for f in direction_probe_results.csv direction_probe_folds.csv source_correction_results.csv source_correction_folds.csv; do
  echo "===== $f ====="; [[ -f "$O/$f" ]] && column -s, -t "$O/$f" || echo MISSING
done
echo "===== REPORT.md ====="; [[ -f "$O/REPORT.md" ]] && cat "$O/REPORT.md" || echo MISSING

