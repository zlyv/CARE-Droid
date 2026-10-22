#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
O=outputs/care-code-function-benign-audit-v1
for f in final_decision.json audit_results.json baseline_integrity_check.json; do
  echo "===== $f ====="
  cat "$O/$f"
done
echo "===== fold_results.csv ====="
column -s, -t "$O/fold_results.csv" || cat "$O/fold_results.csv"
echo "===== leave_one_family_out.csv ====="
column -s, -t "$O/leave_one_family_out.csv" || cat "$O/leave_one_family_out.csv"
