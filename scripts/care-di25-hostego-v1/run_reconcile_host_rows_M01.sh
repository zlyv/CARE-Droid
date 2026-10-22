#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python

# Bookkeeping only: no model construction, fitting, or inference.
"$PY" scripts/care-di25-hostego-v1/reconcile_host_rows_M01.py
