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

V4_ROOT="scripts/run_care_rolegraph_conditioned_dualstream_v4.py"
V4_SUBDIR="scripts/care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py"
if [[ ! -f "$V4_ROOT" && ! -f "$V4_SUBDIR" ]]; then
  echo "Missing CARE v4 runner. Checked both:" >&2
  echo "  $V4_ROOT" >&2
  echo "  $V4_SUBDIR" >&2
  exit 1
fi

REGION_CACHE="artifacts/care-local-region-cache-v1/regions_2017_2021.npz"
if [[ -f "$REGION_CACHE" ]]; then
  echo "Reusing local-region cache: $REGION_CACHE"
else
  echo "Combined local-region cache not found; the runner will build or reuse its strict 2017 source-only cache."
fi

export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-8}"

"$PYTHON_EXE" -c "import numpy, sklearn, torch; print('Dependencies:', numpy.__version__, sklearn.__version__, torch.__version__); print('Device:', 'cuda' if torch.cuda.is_available() else 'cpu')"
"$PYTHON_EXE" scripts/care-malben-contrast-v2/selftest_prepare_inputs.py
"$PYTHON_EXE" scripts/care-malben-contrast-v2/prepare_frozen_inputs.py
"$PYTHON_EXE" scripts/care-malben-contrast-v2/selftest_role_prototypes.py
"$PYTHON_EXE" scripts/care-malben-contrast-v2/selftest_model_contract.py
"$PYTHON_EXE" scripts/care-malben-contrast-v2/selftest_analysis.py

echo "Running strict 2017 OOF role-conditioned evidence ablation."
echo "Future rows are not loaded. Primary router is frozen A25; DI25 is sensitivity only."

"$PYTHON_EXE" scripts/care-malben-contrast-v2/run_care_malben_role_source_v2.py \
  --variants BenignOnly,MaliciousOnly,MalBenContrast \
  --seeds 0,1,2 \
  --max-delta 2.25 \
  --device auto \
  --output artifacts/care-malben-contrast-v2/source/model

"$PYTHON_EXE" scripts/care-malben-contrast-v2/analyze_source_v2.py

cat artifacts/care-malben-contrast-v2/source/source_summary.txt

echo "Source-only experiment complete."
echo "Do not run future diagnostics unless source_decision.json says advance=true."
