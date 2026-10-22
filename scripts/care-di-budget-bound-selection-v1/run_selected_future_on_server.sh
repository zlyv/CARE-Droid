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

ROOT="${OUTPUT_ROOT:-artifacts/care-di-budget-bound-selection-v1}"
CONFIG="$ROOT/source_selection/selected_source_config.json"
FUTURE_RUNNER="scripts/care-rgcondattn-dedup-bounded-future-v1/run_care_rgcondattn_dedup_bounded_future_v1.py"

if [[ ! -f "$CONFIG" ]]; then
  echo "Run run_source_grid_on_server.sh first; selected config is missing." >&2
  exit 1
fi
if [[ ! -f "$FUTURE_RUNNER" ]]; then
  echo "Missing established deduplicated CARE future runner: $FUTURE_RUNNER" >&2
  exit 1
fi

CAP=$("$PYTHON_EXE" -c "import json; print(json.load(open('$CONFIG'))['source_selected_within_ceiling']['max_delta'])")
CAP_TAG=$("$PYTHON_EXE" -c 'import sys; v=float(sys.argv[1]); s=("%.6f" % v).rstrip("0").rstrip("."); print("max_delta_" + s.replace("-", "m").replace(".", "p"))' "$CAP")
OUT="$ROOT/future_model_$CAP_TAG"

export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "[future diagnostic] using source-frozen max_delta=$CAP"
"$PYTHON_EXE" "$FUTURE_RUNNER" \
  --years 2018,2019,2020,2021 \
  --variants RGConcat,RGCondAttn \
  --seeds 0,1,2 \
  --max-delta "$CAP" \
  --device auto \
  --output "$OUT"

"$PYTHON_EXE" scripts/care-di-budget-bound-selection-v1/evaluate_selected_future.py \
  --selected-config "$CONFIG" \
  --future-predictions "$OUT/future_predictions_diagnostic.csv" \
  --output "$ROOT/future_diagnostic"

cat "$ROOT/future_diagnostic/REPORT.txt"
