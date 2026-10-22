#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid

PACKAGE="scripts/care-route-stage2-within-year-matrix-v1"
ROOT="artifacts/care-route-stage2-within-year-matrix-v1"
ARMS="$ROOT/arms"
FUTURE_LINEAGE="${CARE_FUTURE_LINEAGE:-artifacts/care-future-local-assets-v2/future_lineage_2018_2021.csv}"
FUTURE_COMPACT_ROOT="${CARE_FUTURE_COMPACT_ROOT:-/mnt/HDD1/boyuanjie/tasa-droid-storage/care-route-stage2-future-matrix-v1/compact_future_2018_2021_fixed3}"
mkdir -p "$ARMS" "$ROOT/shared"
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ML_PYTHON="${ML_PYTHON_BIN:-${PYTHON_BIN:-/home/boyuanjie/envs/tif/bin/python}}"
if [[ ! -x "$ML_PYTHON" ]] || ! "$ML_PYTHON" -c "import numpy, sklearn, torch, yaml" >/dev/null 2>&1; then
  echo "ML/Androguard interpreter dependencies are missing: $ML_PYTHON" >&2
  exit 1
fi
ROUTER_PYTHON="${ROUTER_PYTHON_BIN:-}"
if [[ -z "$ROUTER_PYTHON" ]]; then
  for candidate in /home/boyuanjie/venvs/malscan/bin/python /home/boyuanjie/envs/tif/bin/python "$ML_PYTHON"; do
    [[ -x "$candidate" ]] || continue
    if "$candidate" -c "import numpy, sklearn, torch, xgboost" >/dev/null 2>&1; then
      ROUTER_PYTHON="$candidate"; break
    fi
  done
fi
if [[ -z "$ROUTER_PYTHON" ]]; then
  echo "No existing XGBoost router interpreter was found." >&2; exit 1
fi

echo "ML Python: $ML_PYTHON"
"$ML_PYTHON" -c "import numpy, sklearn, torch; print('ML deps:', numpy.__version__, sklearn.__version__, torch.__version__, 'cuda='+str(torch.cuda.is_available()))"
echo "Router Python: $ROUTER_PYTHON"
"$ROUTER_PYTHON" -c "import numpy, sklearn, xgboost; print('Router deps:', numpy.__version__, sklearn.__version__, xgboost.__version__)"

"$ML_PYTHON" "$PACKAGE/selftest_within_year_matrix.py"
"$ML_PYTHON" "$PACKAGE/ensure_future_compact_features.py" \
  --lineage "$FUTURE_LINEAGE" --output "$FUTURE_COMPACT_ROOT" \
  --extractor "$PACKAGE/vendor/run_compact_feature_extraction_ag.py" \
  --mapping "$PACKAGE/configs/semantic_mapping_v1.yaml" \
  --priority "$PACKAGE/configs/sensitive_endpoint_priority_v1.yaml"
"$ML_PYTHON" "$PACKAGE/prepare_frozen_inputs.py" \
  --future-lineage "$FUTURE_LINEAGE" \
  --future-features "$FUTURE_COMPACT_ROOT/features.csv" \
  --future-schema "$FUTURE_COMPACT_ROOT/feature_schema.json" \
  --output "$ROOT/preflight"

ROUTER_INPUTS="$ROOT/shared/within_year_router_inputs.csv"
ROUTER_PROTOCOL="$ROOT/shared/router_protocol.json"
if [[ ! -f "$ROUTER_INPUTS" ]] || [[ "$(($(wc -l < "$ROUTER_INPUTS") - 1))" -ne 1989 ]] || \
   [[ ! -f "$ROUTER_PROTOCOL" ]] || \
   ! grep -q '"feature_contract_version": "source-frozen-compact76-foldsafe-v1"' "$ROUTER_PROTOCOL"; then
  "$ROUTER_PYTHON" "$PACKAGE/build_within_year_router_inputs.py" \
    --resolved "$ROOT/preflight/resolved_inputs.json" --output "$ROUTER_INPUTS"
else
  echo "[resume] complete within-year router OOF file exists"
fi

run_arm() {
  local representation="$1"
  local head="$2"
  local output="$ARMS/${representation}_${head}"
  local prediction="$output/within_year_expert_predictions.csv"
  local protocol="$output/protocol.json"
  if [[ -f "$prediction" ]] && [[ "$(($(wc -l < "$prediction") - 1))" -eq 1989 ]] && \
     [[ -f "$protocol" ]] && \
     grep -q '"feature_contract_version": "source-frozen-compact76-foldsafe-v1"' "$protocol"; then
    echo "[resume] complete within-year arm exists: $representation $head"
    return
  fi
  mkdir -p "$output"
  "$ML_PYTHON" "$PACKAGE/run_within_year_arm.py" \
    --representation "$representation" --head "$head" \
    --router-inputs "$ROUTER_INPUTS" \
    --resolved "$ROOT/preflight/resolved_inputs.json" \
    --future-inter-cache "artifacts/care-route-stage2-future-matrix-v1/shared/future_interz.npz" \
    --device auto --max-delta 2.25 --output "$output"
}

run_arm anomaly direct
run_arm anomaly bounded
run_arm malben direct
run_arm malben bounded

"$ML_PYTHON" "$PACKAGE/summarize_within_year_matrix.py" \
  --router-inputs "$ROUTER_INPUTS" --arms-root "$ARMS" --output "$ROOT/results"
if [[ -f "artifacts/care-route-stage2-future-matrix-v1/results/future_annual_metrics.csv" ]]; then
  "$ML_PYTHON" "$PACKAGE/compare_temporal_gap.py" \
    --same-year "$ROOT/results/within_year_annual_metrics.csv" \
    --temporal "artifacts/care-route-stage2-future-matrix-v1/results/future_annual_metrics.csv" \
    --output "$ROOT/results"
fi
cat "$ROOT/results/REPORT.txt"
echo "WITHIN_YEAR_MATRIX_DONE: $ROOT/results"
