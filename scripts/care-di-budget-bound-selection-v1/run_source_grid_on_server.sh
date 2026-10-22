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

CAPS="${CAPS:-0.25,0.5,0.75,1.0,1.25,1.5,2.0}"
BUDGETS="${BUDGETS:-0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50}"
ROOT="${OUTPUT_ROOT:-artifacts/care-di-budget-bound-selection-v1}"
SOURCE_RUNNER="scripts/care-rgcondattn-dedup-bounded-source-v1/run_care_rgcondattn_dedup_bounded_source_v1.py"

if [[ ! -f "$SOURCE_RUNNER" ]]; then
  echo "Missing established deduplicated CARE source runner: $SOURCE_RUNNER" >&2
  exit 1
fi

export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"$PYTHON_EXE" scripts/care-di-budget-bound-selection-v1/selftest_selection.py

IFS=',' read -r -a CAP_ARRAY <<< "$CAPS"
for cap in "${CAP_ARRAY[@]}"; do
  tag=$("$PYTHON_EXE" -c 'import sys; v=float(sys.argv[1]); s=("%.6f" % v).rstrip("0").rstrip("."); print("max_delta_" + s.replace("-", "m").replace(".", "p"))' "$cap")
  out="$ROOT/retrained_source/$tag"
  echo "[source grid] max_delta=$cap -> $out"
  "$PYTHON_EXE" "$SOURCE_RUNNER" \
    --variants RGCondAttn \
    --seeds 0,1,2 \
    --max-delta "$cap" \
    --device auto \
    --output "$out"
  "$PYTHON_EXE" scripts/care-di-budget-bound-selection-v1/stamp_retrain_metadata.py \
    --output "$out" \
    --max-delta "$cap"
done

"$PYTHON_EXE" scripts/care-di-budget-bound-selection-v1/collect_source_grid.py \
  --retrain-root "$ROOT/retrained_source" \
  --caps "$CAPS" \
  --budgets "$BUDGETS" \
  --budget-ceiling 0.25 \
  --output "$ROOT/source_selection"

cat "$ROOT/source_selection/REPORT.txt"
echo "Source selection complete. Freeze selected_source_config.json before future diagnostics."
