#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python
"$PY" scripts/care-di25-hostego-dmd-bounded-v1/analyze_keep_collapse.py \
  2>&1 | tee artifacts/care-di25-hostego-dmd-bounded-v1.keep-collapse-posthoc.log

