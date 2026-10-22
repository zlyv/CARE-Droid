#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${CARE_ML_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
[[ -x "$PY" ]] || { echo "BLOCKED_BY_CONTRACT: missing $PY" >&2; exit 2; }
"$PY" scripts/care-di25-noego-safe-authority-full-v1/selftest.py
MATRIX=scripts/care-route-stage2-future-matrix-v1
MATRIX_ROOT=artifacts/care-route-stage2-future-matrix-v1
PREFLIGHT=$MATRIX_ROOT/preflight
FUTURE_LINEAGE=artifacts/care-di25-noego-safe-authority-full-v1/preflight/future_lineage_2018_2021_exact.csv
FUTURE_ASSET_ROOT=/mnt/HDD1/boyuanjie/tasa-droid-storage/care-future-local-assets-v2
FUTURE_SHARDS=$FUTURE_ASSET_ROOT/shards
FUTURE_REGIONS=$FUTURE_ASSET_ROOT/regions_2018_2021.npz
FUTURE_ENDPOINT=$FUTURE_ASSET_ROOT/endpoint_centered_tensor_cache_2018_2021.jsonl.gz
FUTURE_MANIFEST=/mnt/HDD1/boyuanjie/tasa-droid-storage/datasets/android/AndroZoo/features/v12_selective_role_v3_full_2017_2021/temporal_eval_manifest.csv
FUTURE_STATUS=/mnt/HDD1/boyuanjie/tasa-droid-storage/datasets/android/AndroZoo/features/v12_selective_role_v3_full_2017_2021/v12_full_extraction_status.csv
FUTURE_COMPACT_SOURCE=/mnt/HDD1/boyuanjie/tasa-droid-storage/care-route-stage2-future-matrix-v1/compact_future_2018_2021_fixed3
FUTURE_COMPACT_ROOT=/mnt/HDD1/boyuanjie/tasa-droid-storage/care-route-stage2-future-matrix-v1/compact_future_2018_2021_lineage_exact_v5
mkdir -p "$PREFLIGHT"
for required in "$FUTURE_MANIFEST" "$FUTURE_STATUS"; do
  [[ -s "$required" ]] || { echo "BLOCKED_BY_CONTRACT: missing/empty lineage authority $required" >&2; exit 2; }
done
[[ -d "$FUTURE_SHARDS" ]] || { echo "BLOCKED_BY_CONTRACT: missing frozen shard directory $FUTURE_SHARDS" >&2; exit 2; }

echo "[repair] rebuilding exact future lineage with the 1,989 shards as cohort authority"
"$PY" scripts/care-di25-noego-safe-authority-full-v1/rebuild_future_lineage_from_shards.py \
  --manifest "$FUTURE_MANIFEST" --status "$FUTURE_STATUS" --shards "$FUTURE_SHARDS" \
  --storage-root /mnt/HDD1/boyuanjie/tasa-droid-storage \
  --output "$FUTURE_LINEAGE" \
  --audit artifacts/care-di25-noego-safe-authority-full-v1/preflight/future_lineage_reconciliation.json

echo "[repair] rebuilding merged assets from the 1,989 frozen shards; no APK is parsed"
"$PY" scripts/care-di25-noego-safe-authority-full-v1/rebuild_future_assets_from_shards.py \
  --lineage "$FUTURE_LINEAGE" --shards "$FUTURE_SHARDS" \
  --output "$FUTURE_ASSET_ROOT" \
  --audit artifacts/care-di25-noego-safe-authority-full-v1/preflight/future_asset_rebuild.json

echo "[repair] running complete post-merge hard gate"
"$PY" scripts/care-di25-noego-safe-authority-full-v1/hard_gate_rebuilt_future_assets.py \
  --lineage "$FUTURE_LINEAGE" --root "$FUTURE_ASSET_ROOT" \
  --audit artifacts/care-di25-noego-safe-authority-full-v1/preflight/future_asset_hard_gate.json
echo "[prepare] rebuilding exact label-sealed future input manifest"
"$PY" scripts/care-di25-noego-safe-authority-full-v1/sanitize_future_compact.py \
  --lineage "$FUTURE_LINEAGE" --source "$FUTURE_COMPACT_SOURCE" --output "$FUTURE_COMPACT_ROOT"
"$PY" "$MATRIX/ensure_future_compact_features.py" \
  --lineage "$FUTURE_LINEAGE" --output "$FUTURE_COMPACT_ROOT" \
  --extractor "$MATRIX/vendor/run_compact_feature_extraction_ag.py" \
  --mapping "$MATRIX/configs/semantic_mapping_v1.yaml" \
  --priority "$MATRIX/configs/sensitive_endpoint_priority_v1.yaml"
"$PY" "$MATRIX/prepare_frozen_inputs.py" \
  --future-lineage "$FUTURE_LINEAGE" \
  --future-features "$FUTURE_COMPACT_ROOT/features.csv" \
  --future-schema "$FUTURE_COMPACT_ROOT/feature_schema.json" \
  --output "$PREFLIGHT"
SOURCE_SCORE="$PREFLIGHT/source_sample_scores.csv"
FUTURE_ROUTER="$PREFLIGHT/future_router_inputs_label_free.csv"
source_n=0; future_n=0
[[ -s "$SOURCE_SCORE" ]] && source_n=$(($(wc -l < "$SOURCE_SCORE")-1))
[[ -s "$FUTURE_ROUTER" ]] && future_n=$(($(wc -l < "$FUTURE_ROUTER")-1))
if [[ "$source_n" -ne 799 || "$future_n" -ne 1989 ]]; then
  ROUTER_PY="${ROUTER_PYTHON_BIN:-}"
  if [[ -z "$ROUTER_PY" ]]; then
    for candidate in /home/boyuanjie/venvs/malscan/bin/python /home/boyuanjie/envs/tif/bin/python "$(command -v python3 2>/dev/null || true)"; do
      [[ -n "$candidate" && -x "$candidate" ]] || continue
      if "$candidate" -c "import numpy,sklearn,torch,xgboost" >/dev/null 2>&1; then ROUTER_PY="$candidate";break;fi
    done
  fi
  [[ -n "$ROUTER_PY" ]] || { echo "BLOCKED_BY_CONTRACT: router inputs are missing and no existing Python contains xgboost" >&2;exit 2; }
  "$ROUTER_PY" "$MATRIX/build_frozen_router_inputs.py" --resolved "$PREFLIGHT/resolved_inputs.json"
fi
"$PY" scripts/care-di25-noego-safe-authority-full-v1/preflight.py
"$PY" scripts/care-di25-noego-safe-authority-full-v1/run_nested_source.py
"$PY" scripts/care-di25-noego-safe-authority-full-v1/run_frozen_future.py \
  --future-regions "$FUTURE_REGIONS" --future-cache "$FUTURE_ENDPOINT"
"$PY" scripts/care-di25-noego-safe-authority-full-v1/summarize_future.py
