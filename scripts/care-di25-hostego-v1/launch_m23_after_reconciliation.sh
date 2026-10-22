#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python

# M2/M3 is launched only when the bookkeeping gate exits successfully.
"$PY" scripts/care-di25-hostego-v1/reconcile_host_rows_M01.py
grep -q '^HOST_ROW_COUNT_RECONCILIATION_PASS$' \
  artifacts/care-di25-hostego-v1/audits/host_row_count_reconciliation.txt
bash scripts/care-di25-hostego-v1/launch_tmux.sh m23
