#!/usr/bin/env bash
set -euo pipefail
OUT=/home/boyuanjie/projects/tasa-droid/outputs/care_role_host_aligned_fusion_v1
for f in final_decision.json baseline_integrity_check.json REPORT.md; do echo "===== $f ====="; test -f "$OUT/$f" && cat "$OUT/$f" || echo MISSING; done
