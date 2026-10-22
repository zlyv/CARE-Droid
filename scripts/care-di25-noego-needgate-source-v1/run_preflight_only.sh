#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/boyuanjie/projects/tasa-droid"
PACKAGE="${PROJECT_ROOT}/scripts/care-di25-noego-needgate-source-v1"
PYTHON="${CARE_ML_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"

cd "${PROJECT_ROOT}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "BLOCKED_BY_CONTRACT: missing executable Python: ${PYTHON}" >&2
  exit 2
fi

"${PYTHON}" "${PACKAGE}/selftest.py"
"${PYTHON}" "${PACKAGE}/preflight.py"

echo "SOURCE_NEEDGATE_SELFTEST_AND_PREFLIGHT_ONLY_DONE"
echo "formal_training_started = false"
