#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python
ROOT=artifacts/care-di25-hostego-v1

"$PY" scripts/care-di25-hostego-v1/preflight.py
"$PY" scripts/care-di25-hostego-v1/selftest_hostego.py
"$PY" scripts/care-di25-hostego-v1/smoke_hostego.py

PHASE="${1:-m01}"
if [[ "$PHASE" == "m01" ]]; then
  "$PY" scripts/care-di25-hostego-v1/run_care_di25_hostego_v1.py --arm M0 --seeds 0,1,2 --device auto
  "$PY" scripts/care-di25-hostego-v1/run_care_di25_hostego_v1.py --arm M1 --seeds 0,1,2 --device auto
  "$PY" scripts/care-di25-hostego-v1/audit_reproduction.py --arms m01
  "$PY" scripts/care-di25-hostego-v1/summarize_m01.py
  echo "M0/M1 finished. Inspect their source_summary.csv files before launching m23."
elif [[ "$PHASE" == "m23" ]]; then
  [[ -f "$ROOT/source/M0_host_anomaly_bounded/source_predictions.csv" ]]
  [[ -f "$ROOT/source/M1_hostego_anomaly_bounded/source_predictions.csv" ]]
  "$PY" scripts/care-di25-hostego-v1/run_care_di25_hostego_v1.py --arm M2 --seeds 0,1,2 --device auto
  "$PY" scripts/care-di25-hostego-v1/run_care_di25_hostego_v1.py --arm M3 --seeds 0,1,2 --device auto
  "$PY" scripts/care-di25-hostego-v1/audit_reproduction.py
  "$PY" scripts/care-di25-hostego-v1/summarize_hostego.py
else
  echo "Usage: $0 [m01|m23]" >&2; exit 2
fi
