#!/usr/bin/env bash
set -euo pipefail
OUT="${1:?usage: collect_results.sh output-dir}"
for f in feasibility_decision.json coverage_audit.json method_metadata_coverage.json baseline_integrity_check.json report.md;do echo "===== $f =====";[[ -s "$OUT/$f" ]]||{ echo MISSING;continue;};case "$f" in *.json) python3 -m json.tool "$OUT/$f";;*) cat "$OUT/$f";;esac;done

