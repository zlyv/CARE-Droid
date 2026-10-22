#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; OUT="$ROOT/outputs/care-tb-frs-source-v1"
for f in final_decision.json baseline_integrity_check.json REPORT.md; do if [[ -f "$OUT/$f" ]]; then echo "===== $f ====="; cat "$OUT/$f"; else echo "MISSING: $OUT/$f"; fi; done
if [[ -f "$OUT/fold_results.csv" ]]; then echo "===== fold_results.csv ====="; column -s, -t "$OUT/fold_results.csv" 2>/dev/null || cat "$OUT/fold_results.csv"; fi
