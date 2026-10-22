#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
CODE="scripts/care-litesemantic-noshortcut-source-v1"
ROOT="artifacts/care-litesemantic-noshortcut-source-v1"
mkdir -p "$ROOT"
PY=""
for candidate in /home/boyuanjie/envs/tif/bin/python /home/boyuanjie/venvs/malscan/bin/python; do
  if [[ -x "$candidate" ]] && "$candidate" -c 'import androguard,yaml,numpy,scipy,sklearn,xgboost' >/dev/null 2>&1; then PY="$candidate"; break; fi
done
if [[ -z "$PY" ]]; then echo "BLOCKED_BY_CONTRACT: no existing Python has androguard/yaml/numpy/scipy/sklearn/xgboost"; echo "No dependency was installed."; exit 2; fi
echo "Python: $PY"
"$PY" "$CODE/selftest.py"
"$PY" "$CODE/preflight.py"
"$PY" "$CODE/extract_features.py"
"$PY" "$CODE/run_source_oof.py"
echo "LITESEMANTIC_NOSHORTCUT_FULL_RUN_DONE"
