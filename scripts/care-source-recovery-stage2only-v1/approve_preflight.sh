#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
ROOT="artifacts/care-source-recovery-stage2only-v1/preflight"
PY="${PYTHON_BIN:-/home/boyuanjie/envs/tif/bin/python}"
cat "$ROOT/source_lineage_integrity_audit.txt"
cat "$ROOT/ego_smoke_test_summary.txt"
"$PY" scripts/care-source-recovery-stage2only-v1/preflight_approval.py approve --root "$ROOT"
echo "The exact two inspected gates are now approved."
