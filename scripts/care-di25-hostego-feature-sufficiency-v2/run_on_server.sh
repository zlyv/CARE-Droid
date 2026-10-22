#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python
OUT=artifacts/care-di25-hostego-dmd-bounded-v1/feature_sufficiency_audit
mkdir -p "$OUT"

"$PY" scripts/care-di25-hostego-feature-sufficiency-v2/preflight.py
"$PY" scripts/care-di25-hostego-feature-sufficiency-v2/audit_and_replay_validation.py
"$PY" scripts/care-di25-hostego-feature-sufficiency-v2/reconstruct_and_cache_r4.py
"$PY" scripts/care-di25-hostego-feature-sufficiency-v2/run_strict_probes.py

cat "$OUT/FEATURE_SUFFICIENCY_REPORT_V2.txt"
