#!/usr/bin/env bash
set -euo pipefail
PROJECT="${CSRC_PROJECT:-/home/boyuanjie/projects/tasa-droid}";OUTPUT="${CSRC_OUTPUT:-$PROJECT/outputs/care-svm-c43-remove-bounded-correction-v1}"
for f in final_decision.json source_results.csv paired_vs_bounded.csv source_fold_results.csv raw_output_audit.csv baseline_integrity_check.json REPORT.md;do echo "===== $f =====";cat "$OUTPUT/$f";done
