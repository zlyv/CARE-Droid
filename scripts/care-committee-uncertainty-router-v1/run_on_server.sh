#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

PYTHON_EXE=""
if [[ -n "${PYTHON_BIN:-}" ]]; then
  CANDIDATES=("${PYTHON_BIN}")
else
  # The completed full MalScan experiment uses this environment and therefore
  # it is the most reliable existing source of the required XGBoost package.
  CANDIDATES=(
    /home/boyuanjie/venvs/malscan/bin/python
    /home/boyuanjie/envs/malscan/bin/python
    /home/boyuanjie/envs/tif/bin/python
  )
  if command -v python3 >/dev/null 2>&1; then
    CANDIDATES+=("$(command -v python3)")
  fi
fi

for candidate in "${CANDIDATES[@]}"; do
  if [[ -x "${candidate}" ]] && "${candidate}" -c \
    "import numpy, sklearn, xgboost, torch" >/dev/null 2>&1; then
    PYTHON_EXE="${candidate}"
    break
  fi
done

if [[ -z "${PYTHON_EXE}" ]]; then
  echo "No existing Python environment contains numpy, sklearn, xgboost, and torch." >&2
  echo "Expected MalScan environment: /home/boyuanjie/venvs/malscan/bin/python" >&2
  echo "No package was installed or upgraded. Set PYTHON_BIN to a compatible interpreter." >&2
  exit 1
fi

echo "Using Python: ${PYTHON_EXE}"
"${PYTHON_EXE}" -c "import sklearn, xgboost, torch; print('Dependency check passed: sklearn=' + sklearn.__version__ + ' xgboost=' + xgboost.__version__ + ' torch=' + torch.__version__)"
"${PYTHON_EXE}" scripts/care-committee-uncertainty-router-v1/selftest_committee_uncertainty.py

mkdir -p artifacts/care-committee-uncertainty-router-source-v1
mkdir -p artifacts/care-committee-uncertainty-router-future-v1

"${PYTHON_EXE}" scripts/care-committee-uncertainty-router-v1/run_committee_source_v1.py
"${PYTHON_EXE}" scripts/care-committee-uncertainty-router-v1/run_committee_future_v1.py

echo "Committee source and future diagnostics completed."
echo "Future years are diagnostic-only and were not used for model or router selection."
