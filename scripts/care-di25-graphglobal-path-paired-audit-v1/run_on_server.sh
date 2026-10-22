#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python

"$PY" scripts/care-di25-graphglobal-path-paired-audit-v1/selftest.py
"$PY" scripts/care-di25-graphglobal-path-paired-audit-v1/preflight.py
"$PY" scripts/care-di25-graphglobal-path-paired-audit-v1/run_paired_audit.py \
  --bootstrap-iterations 10000 --bootstrap-seed 20260923

