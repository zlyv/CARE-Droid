#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
CODE="scripts/care-litelocal-stage1-attribution-source-v1"
OUT="artifacts/care-litelocal-stage1-attribution-source-v1"
mkdir -p "$OUT"
PY=""
for candidate in /home/boyuanjie/envs/tif/bin/python /home/boyuanjie/venvs/malscan/bin/python; do
  if [[ -x "$candidate" ]] && "$candidate" -c 'import numpy,scipy,sklearn,xgboost' >/dev/null 2>&1; then PY="$candidate"; break; fi
done
if [[ -z "$PY" ]]; then
  echo "BLOCKED_BY_CONTRACT: no existing Python has numpy/scipy/sklearn/xgboost"
  echo "No dependency was installed."
  exit 2
fi
echo "Python: $PY"
"$PY" "$CODE/selftest.py"
"$PY" "$CODE/preflight.py"
"$PY" "$CODE/run_audit.py"
echo "LITELOCAL_STAGE1_ATTRIBUTION_FULL_RUN_DONE"
