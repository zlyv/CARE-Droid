#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid

PACKAGE="scripts/care-route-stage2-future-matrix-v1"
ROOT="artifacts/care-route-stage2-future-matrix-v1"
ARMS="$ROOT/arms"
mkdir -p "$ARMS" "$ROOT/shared"
FUTURE_LINEAGE="${CARE_FUTURE_LINEAGE:-artifacts/care-future-local-assets-v2/future_lineage_2018_2021.csv}"
FUTURE_COMPACT_ROOT="${CARE_FUTURE_COMPACT_ROOT:-/mnt/HDD1/boyuanjie/tasa-droid-storage/care-route-stage2-future-matrix-v1/compact_future_2018_2021_fixed3}"
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ML_PYTHON="${ML_PYTHON_BIN:-${PYTHON_BIN:-/home/boyuanjie/envs/tif/bin/python}}"
if [[ ! -x "$ML_PYTHON" ]] || ! "$ML_PYTHON" -c "import numpy, sklearn, torch" >/dev/null 2>&1; then
  echo "ML interpreter is missing NumPy/scikit-learn/PyTorch: $ML_PYTHON" >&2
  echo "Set ML_PYTHON_BIN to the CUDA CARE environment." >&2
  exit 1
fi

ROUTER_PYTHON="${ROUTER_PYTHON_BIN:-}"
if [[ -z "$ROUTER_PYTHON" ]]; then
  for candidate in \
    /home/boyuanjie/venvs/malscan/bin/python \
    /home/boyuanjie/envs/tif/bin/python \
    "$ML_PYTHON" \
    "$(command -v python3 2>/dev/null || true)"; do
    [[ -n "$candidate" && -x "$candidate" ]] || continue
    if "$candidate" -c "import numpy, sklearn, torch, xgboost" >/dev/null 2>&1; then
      ROUTER_PYTHON="$candidate"
      break
    fi
  done
fi
if [[ -z "$ROUTER_PYTHON" ]] || ! "$ROUTER_PYTHON" -c "import numpy, sklearn, torch, xgboost" >/dev/null 2>&1; then
  echo "No existing interpreter contains NumPy/scikit-learn/PyTorch/XGBoost." >&2
  echo "Set ROUTER_PYTHON_BIN to the interpreter used by the completed committee experiment." >&2
  exit 1
fi

echo "ML Python: $ML_PYTHON"
"$ML_PYTHON" -c "import numpy, sklearn, torch; print('ML deps:', numpy.__version__, sklearn.__version__, torch.__version__, 'cuda='+str(torch.cuda.is_available()))"
echo "Router Python: $ROUTER_PYTHON"
"$ROUTER_PYTHON" -c "import numpy, sklearn, torch, xgboost; print('Router deps:', numpy.__version__, sklearn.__version__, torch.__version__, xgboost.__version__)"

"$ML_PYTHON" "$PACKAGE/selftest_future_matrix.py"

# Host/Ego caches already exist. Only the previously unpersisted Compact76
# rows are extracted here. The dedicated HDD cache is append-resumable and is
# protected by exact-lineage, duplicate, failure and outside-row hard gates.
"$ML_PYTHON" "$PACKAGE/ensure_future_compact_features.py" \
  --lineage "$FUTURE_LINEAGE" \
  --output "$FUTURE_COMPACT_ROOT" \
  --extractor "$PACKAGE/vendor/run_compact_feature_extraction_ag.py" \
  --mapping "$PACKAGE/configs/semantic_mapping_v1.yaml" \
  --priority "$PACKAGE/configs/sensitive_endpoint_priority_v1.yaml"

"$ML_PYTHON" "$PACKAGE/prepare_frozen_inputs.py" \
  --future-lineage "$FUTURE_LINEAGE" \
  --future-features "$FUTURE_COMPACT_ROOT/features.csv" \
  --future-schema "$FUTURE_COMPACT_ROOT/feature_schema.json" \
  --output "$ROOT/preflight"
"$ROUTER_PYTHON" "$PACKAGE/build_frozen_router_inputs.py" \
  --resolved "$ROOT/preflight/resolved_inputs.json"

run_arm() {
  local representation="$1"
  local head="$2"
  local output="$ARMS/${representation}_${head}"
  local prediction="$output/future_expert_predictions.csv"
  if [[ -f "$prediction" ]] && [[ "$(($(wc -l < "$prediction") - 1))" -eq 1989 ]]; then
    echo "[resume] complete arm exists: $representation $head"
    return
  fi
  mkdir -p "$output"
  "$ML_PYTHON" "$PACKAGE/run_future_arm.py" \
    --representation "$representation" --head "$head" \
    --device auto --max-delta 2.25 \
    --output "$output"
}

run_arm anomaly direct
run_arm anomaly bounded
run_arm malben direct
run_arm malben bounded

"$ML_PYTHON" "$PACKAGE/summarize_matrix.py" \
  --arms-root "$ARMS" --output "$ROOT/results"
cat "$ROOT/results/REPORT.txt"
echo "FUTURE_MATRIX_DONE: $ROOT/results"
