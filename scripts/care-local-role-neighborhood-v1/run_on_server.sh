#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python
S=scripts/care-local-role-neighborhood-v1
A=artifacts/care-local-role-neighborhood-v1
mkdir -p "$A"
export PYTHONWARNINGS=ignore
export LOGURU_LEVEL=ERROR
"$PY" "$S/selftest.py"
"$PY" "$S/audit_existing.py"
if "$PY" - <<'PY'
import json
d=json.load(open('artifacts/care-local-role-neighborhood-v1/existing_experiment_audit.json'))
raise SystemExit(0 if d['status']=='FOUND' and d['equivalent'] else 1)
PY
then
  echo LOCAL_ROLE_NEIGHBOR_REUSE_EXISTING_RESULT
  exit 0
fi
"$PY" "$S/write_protocol.py"
"$PY" "$S/preflight.py"
"$PY" "$S/run_probe.py"
"$PY" "$S/summarize.py"
echo CARE_LOCAL_ROLE_NEIGHBOR_SOURCE_PIPELINE_DONE
echo artifacts=$A
echo report=$A/REPORT.txt
echo final_summary=$A/final_summary.json

