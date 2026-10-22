#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid;O=outputs/care-function-role-explanation-v1
for f in final_decision.json probe_results.json baseline_integrity_check.json;do echo "===== $f =====";cat "$O/$f";done
echo "===== fold_probe_results.csv =====";column -s, -t "$O/fold_probe_results.csv" || cat "$O/fold_probe_results.csv"
