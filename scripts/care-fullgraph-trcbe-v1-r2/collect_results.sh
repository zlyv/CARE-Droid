#!/usr/bin/env bash
set -euo pipefail
PROJECT="${FULLGRAPH_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${FULLGRAPH_OUTPUT:-$PROJECT/outputs/care-fullgraph-trcbe-v2}"
for file in feasibility_decision.json pilot_decision.json trcbe_fullgraph_decision.json final_decision.json baseline_integrity_check.json; do
  if [[ -f "$OUTPUT/$file" ]]; then echo "===== $file ====="; cat "$OUTPUT/$file"; fi
done
if [[ -f "$OUTPUT/report.md" ]]; then echo "===== report.md ====="; cat "$OUTPUT/report.md"; fi
