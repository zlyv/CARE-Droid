#!/usr/bin/env bash
set -euo pipefail
out="${1:-outputs/care-ccbc-source-v1}"
echo "===== FINAL SUMMARY ====="
cat "$out/final_summary.json"
echo "===== REPORT ====="
cat "$out/report.md"
echo "===== END TO END ====="
python - "$out/end_to_end_results.json" <<'PY'
import json,sys
for r in json.load(open(sys.argv[1]))["aggregate"]:
 print(r["arm"], "BA=%.6f"%r["BA"], "AUC=%.6f"%r["AUC"], "F1=%.6f"%r["F1"], "fixed=%s"%r["fixed"], "broken=%s"%r["broken"], "net=%s"%r["net"], "fixed_FN=%s"%r["fixed_FN"], "fixed_FP=%s"%r["fixed_FP"])
PY
echo "===== SUFFICIENCY ====="
python - "$out/sufficiency_probe.json" <<'PY'
import json,sys
for r in json.load(open(sys.argv[1]))["variants"]:
 print(r["variant"], "FN-vs-TN=%.6f"%r["FN_vs_TN_AUC"], "FP-vs-TP=%.6f"%r["FP_vs_TP_AUC"])
PY
echo "===== INTEGRITY ====="
cat "$out/baseline_integrity_check.json"
