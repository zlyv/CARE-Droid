#!/usr/bin/env bash
set -euo pipefail
PROJECT="${RPF_PROJECT:-/home/boyuanjie/projects/tasa-droid}";OUTPUT="${RPF_OUTPUT:-$PROJECT/outputs/care-direct-c43-rolepath-factorial-v1}"
for f in final_decision.json factorial_effects.json source_results.csv paired_vs_R11.csv source_fold_results.csv factorial_fold_effects.csv baseline_integrity_check.json REPORT.md;do echo "===== $f =====";cat "$OUTPUT/$f";done
