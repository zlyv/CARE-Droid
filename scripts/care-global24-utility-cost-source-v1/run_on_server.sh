#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
CODE="scripts/care-global24-utility-cost-source-v1"
ROOT="artifacts/care-global24-utility-cost-source-v1"
mkdir -p "$ROOT"

PY=""
for candidate in \
  /home/boyuanjie/venvs/malscan/bin/python \
  /home/boyuanjie/envs/tif/bin/python
do
  if [[ -x "$candidate" ]] && "$candidate" -c 'import numpy, sklearn, torch, xgboost' >/dev/null 2>&1; then
    PY="$candidate"
    break
  fi
done
if [[ -z "$PY" ]]; then
  echo "BLOCKED_BY_CONTRACT: no existing Python has numpy/sklearn/torch/xgboost"
  echo "No dependency was installed."
  exit 2
fi
echo "Python: $PY"
"$PY" -c 'import numpy,sklearn,torch,xgboost; print("Dependencies:",numpy.__version__,sklearn.__version__,torch.__version__,xgboost.__version__)'
"$PY" "$CODE/selftest.py"
"$PY" "$CODE/preflight.py"
"$PY" "$CODE/run_audit.py" --jobs 4
echo "GLOBAL24_UTILITY_COST_SOURCE_FULL_RUN_DONE"
