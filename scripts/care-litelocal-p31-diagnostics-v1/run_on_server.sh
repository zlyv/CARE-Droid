#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
CODE="scripts/care-litelocal-p31-diagnostics-v1"
OUT="artifacts/care-litelocal-p31-diagnostics-v1"
mkdir -p "$OUT"
PY=""
for candidate in /home/boyuanjie/envs/tif/bin/python /home/boyuanjie/venvs/malscan/bin/python; do
  if [[ -x "$candidate" ]] && "$candidate" -c 'import androguard,yaml,numpy,scipy,sklearn,xgboost' >/dev/null 2>&1; then PY="$candidate";break;fi
done
if [[ -z "$PY" ]];then echo "BLOCKED_BY_CONTRACT: no existing Python has required dependencies";echo "No dependency was installed.";exit 2;fi
echo "Python: $PY"
"$PY" "$CODE/selftest.py"
"$PY" "$CODE/preflight.py"
"$PY" "$CODE/run_source_shortcut.py"
"$PY" "$CODE/extract_future_p31.py"
"$PY" "$CODE/run_frozen_future.py" > "$OUT/frozen_future_job.log" 2>&1 &
PID_FROZEN=$!
"$PY" "$CODE/run_within_year.py" > "$OUT/within_year_job.log" 2>&1 &
PID_WITHIN=$!
status=0
wait "$PID_FROZEN" || status=1
wait "$PID_WITHIN" || status=1
cat "$OUT/frozen_future_job.log"
cat "$OUT/within_year_job.log"
if [[ "$status" -ne 0 ]];then echo "BLOCKED_BY_CONTRACT: P31 diagnostic job failed";exit 2;fi
"$PY" "$CODE/summarize_frozen_future.py"
"$PY" "$CODE/build_report.py"
echo "LITELOCAL_P31_DIAGNOSTICS_FULL_RUN_DONE"
