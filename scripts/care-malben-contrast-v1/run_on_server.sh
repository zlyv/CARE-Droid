#!/usr/bin/env bash
set -euo pipefail

cd /home/boyuanjie/projects/tasa-droid

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_EXE="${PYTHON_BIN}"
elif [[ -x /home/boyuanjie/envs/tif/bin/python ]]; then
  PYTHON_EXE=/home/boyuanjie/envs/tif/bin/python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_EXE="$(command -v python3)"
else
  echo "No Python interpreter found; set PYTHON_BIN." >&2
  exit 1
fi

export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"$PYTHON_EXE" -c "import numpy, sklearn, torch; print('Device:', 'cuda' if torch.cuda.is_available() else 'cpu')"
"$PYTHON_EXE" scripts/care-malben-contrast-v1/selftest_malben_contrast.py

"$PYTHON_EXE" scripts/care-malben-contrast-v1/run_care_malben_contrast_source_v1.py \
  --variants RGCondAttn \
  --seeds 0,1,2 \
  --max-delta 2.25 \
  --device auto \
  --output artifacts/care-malben-contrast-v1/source

"$PYTHON_EXE" scripts/care-malben-contrast-v1/compare_source.py

