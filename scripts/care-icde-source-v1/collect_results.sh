#!/usr/bin/env bash
set -euo pipefail
OUT="${1:?usage: collect_results.sh /absolute/output/path}"
for name in decision.json feasibility_decision.json environment_inventory.json apk_asset_audit.json baseline_integrity_check.json report.md; do
  echo "===== $name ====="
  if [[ -s "$OUT/$name" ]]; then
    case "$name" in *.json) python3 -m json.tool "$OUT/$name";; *) cat "$OUT/$name";; esac
  else echo "MISSING"; fi
done

