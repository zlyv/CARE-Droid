#!/usr/bin/env bash
set -euo pipefail
PROJECT="${SVMPBE_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${SVMPBE_OUTPUT:-$PROJECT/outputs/care-svm-positive-benign-evidence-v1}"
for name in final_decision.json evidence_probe.json baseline_integrity_check.json; do
  echo "===== $name ====="
  cat "$OUTPUT/$name"
done
echo "===== source_results.csv ====="
column -s, -t "$OUTPUT/source_results.csv" || cat "$OUTPUT/source_results.csv"
echo "===== fold_probe_results.csv ====="
column -s, -t "$OUTPUT/fold_probe_results.csv" || cat "$OUTPUT/fold_probe_results.csv"
echo "===== REPORT.md ====="
cat "$OUTPUT/REPORT.md"

