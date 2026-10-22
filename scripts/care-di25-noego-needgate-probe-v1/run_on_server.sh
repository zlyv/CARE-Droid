#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/boyuanjie/projects/tasa-droid"
PACKAGE="${PROJECT_ROOT}/scripts/care-di25-noego-needgate-probe-v1"
OUTPUT="${PROJECT_ROOT}/artifacts/care-di25-noego-needgate-probe-v1"
PYTHON="${CARE_ML_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
SOURCE_CACHE="${CARE_SOURCE_CACHE:-artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz}"
FUTURE_CACHE="${CARE_FUTURE_CACHE:-artifacts/care-future-local-assets-v2/endpoint_centered_tensor_cache_2018_2021.jsonl.gz}"
FUTURE_REGIONS="${CARE_FUTURE_REGIONS:-artifacts/care-future-local-assets-v2/regions_2018_2021.npz}"
FUTURE_INTER="${CARE_FUTURE_INTER:-artifacts/care-route-stage2-future-matrix-v1/shared/future_interz.npz}"

cd "${PROJECT_ROOT}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "BLOCKED_BY_CONTRACT: missing executable ML Python: ${PYTHON}" >&2
  exit 2
fi
mkdir -p "${OUTPUT}"

"${PYTHON}" "${PACKAGE}/selftest.py"
COMMON=(--output "${OUTPUT}" --source-cache "${SOURCE_CACHE}" --future-cache "${FUTURE_CACHE}" --future-regions "${FUTURE_REGIONS}" --future-inter-cache "${FUTURE_INTER}")
"${PYTHON}" "${PACKAGE}/preflight.py" "${COMMON[@]}"
"${PYTHON}" "${PACKAGE}/run_needgate_probe.py" "${COMMON[@]}" --device auto
"${PYTHON}" "${PACKAGE}/summarize.py" --output "${OUTPUT}"

echo "NOEGO_NEEDGATE_PACKAGE_DONE"
