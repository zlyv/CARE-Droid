#!/usr/bin/env bash
set -euo pipefail
O="${DCFA_OUTPUT:-/home/boyuanjie/projects/tasa-droid/outputs/care-svm-direct-c43-frontier-audit-v1}"
for f in final_decision.json baseline_integrity_check.json report.md;do echo "===== $f =====";cat "$O/$f";done
echo "===== pareto_frontier.csv =====";column -s, -t "$O/pareto_frontier.csv" || cat "$O/pareto_frontier.csv"
