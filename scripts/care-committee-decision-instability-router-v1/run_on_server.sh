#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_EXE="${PYTHON_BIN}"
elif [[ -x /home/boyuanjie/envs/tif/bin/python ]]; then
  PYTHON_EXE=/home/boyuanjie/envs/tif/bin/python
elif [[ -x /home/boyuanjie/venvs/malscan/bin/python ]]; then
  PYTHON_EXE=/home/boyuanjie/venvs/malscan/bin/python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_EXE="$(command -v python3)"
else
  echo "No Python interpreter found. Set PYTHON_BIN explicitly." >&2
  exit 1
fi

export PYTHONNOUSERSITE=1

"${PYTHON_EXE}" -c "import numpy, sklearn; print('Dependency check passed:', sklearn.__version__)"
"${PYTHON_EXE}" scripts/care-committee-decision-instability-router-v1/selftest_decision_instability.py
"${PYTHON_EXE}" scripts/care-committee-decision-instability-router-v1/run_decision_instability_audit_v1.py

echo "Decision-instability routing audit completed."
echo "Future years are post-hoc diagnostic-only and were not used to define U_DI."
