#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/boyuanjie/projects/tasa-droid"
cd "$PROJECT"
PYTHON_BIN="${PYTHON_BIN:-/home/boyuanjie/envs/tif/bin/python}"
PACKAGE="scripts/care-future-local-assets-v2"
SOURCE_PACKAGE="scripts/care-source-local-assets-v2"
RECOVERY="scripts/care-source-recovery-stage2only-v1"
FUTURE_MANIFEST="/mnt/HDD1/boyuanjie/tasa-droid-storage/datasets/android/AndroZoo/features/v12_selective_role_v3_full_2017_2021/temporal_eval_manifest.csv"
FUTURE_STATUS="/mnt/HDD1/boyuanjie/tasa-droid-storage/datasets/android/AndroZoo/features/v12_selective_role_v3_full_2017_2021/v12_full_extraction_status.csv"
FROZEN_REFERENCE="$PACKAGE/assets/future_reference_1989.csv"
HDD_OUTPUT="${CARE_FUTURE_ASSET_ROOT:-/mnt/HDD1/boyuanjie/tasa-droid-storage/care-future-local-assets-v2}"
PROJECT_OUTPUT="artifacts/care-future-local-assets-v2"

export PYTHONNOUSERSITE=1
mkdir -p "$HDD_OUTPUT" artifacts
if [[ -L "$PROJECT_OUTPUT" ]]; then
  [[ "$(readlink -f "$PROJECT_OUTPUT")" == "$(readlink -f "$HDD_OUTPUT")" ]] || {
    echo "Refusing mismatched artifact symlink: $PROJECT_OUTPUT" >&2; exit 1; }
elif [[ -e "$PROJECT_OUTPUT" ]]; then
  echo "Refusing to overwrite existing non-symlink path: $PROJECT_OUTPUT" >&2
  exit 1
else
  ln -s "$HDD_OUTPUT" "$PROJECT_OUTPUT"
fi

BUILD_ARGS=(
  --manifest "$FUTURE_MANIFEST"
  --status "$FUTURE_STATUS"
  --reference "$FROZEN_REFERENCE"
  --storage-root /mnt/HDD1/boyuanjie/tasa-droid-storage
  --output "$PROJECT_OUTPUT/future_lineage_2018_2021.csv"
  --audit "$PROJECT_OUTPUT/future_lineage_gate.json"
)
"$PYTHON_BIN" "$PACKAGE/build_future_lineage_v2.py" "${BUILD_ARGS[@]}"

EXTRACTOR="$RECOVERY/extractor_repo"
"$PYTHON_BIN" "$SOURCE_PACKAGE/extract_source_local_assets_v2.py" \
  --lineage "$PROJECT_OUTPUT/future_lineage_2018_2021.csv" \
  --mapping "$EXTRACTOR/configs/protocol/semantic_mapping_v1.yaml" \
  --priority "$EXTRACTOR/configs/protocol/sensitive_endpoint_priority_v1.yaml" \
  --edge-semantics "$EXTRACTOR/configs/protocol/semantic_edge_mapping_v1.yaml" \
  --extractor-root "$EXTRACTOR" \
  --output "$PROJECT_OUTPUT" \
  --expected 1989 --asset-tag 2018_2021 --cohort-kind future \
  --years 2018,2019,2020,2021 --checkpoint-every 10 --progress-every 5

"$PYTHON_BIN" "$PACKAGE/audit_future_assets_v2.py" \
  --lineage "$PROJECT_OUTPUT/future_lineage_2018_2021.csv" \
  --root "$PROJECT_OUTPUT" --expected 1989

cat "$PROJECT_OUTPUT/extraction_summary.txt"
cat "$PROJECT_OUTPUT/future_asset_gate.txt"
echo "Future extraction completed. No model fitting or future-label tuning was run."
