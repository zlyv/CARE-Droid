#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/boyuanjie/projects/tasa-droid"
cd "$PROJECT"
PYTHON_BIN="${PYTHON_BIN:-/home/boyuanjie/envs/tif/bin/python}"
PACKAGE="scripts/care-source-local-assets-v2"
RECOVERY="scripts/care-source-recovery-stage2only-v1"
LINEAGE="artifacts/care-source-recovery-stage2only-v1/source_lineage_2017.csv"
HDD_OUTPUT="${CARE_LOCAL_ASSET_ROOT:-/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2}"
PROJECT_OUTPUT="artifacts/care-source-local-assets-v2"

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

cp -f "$LINEAGE" "$PROJECT_OUTPUT/source_lineage_2017.csv"
"$PYTHON_BIN" "$PACKAGE/audit_source_lineage_v2.py" \
  --lineage "$PROJECT_OUTPUT/source_lineage_2017.csv" \
  --folds artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv \
  --features artifacts/p1-features/features.csv \
  --output "$PROJECT_OUTPUT/source_lineage_gate.json"
"$PYTHON_BIN" "$PACKAGE/selftest_exclusive_writer.py"
"$PYTHON_BIN" "$PACKAGE/selftest_asset_reconstruction.py"

EXTRACTOR="$RECOVERY/extractor_repo"
"$PYTHON_BIN" "$PACKAGE/extract_source_local_assets_v2.py" \
  --lineage "$PROJECT_OUTPUT/source_lineage_2017.csv" \
  --mapping "$EXTRACTOR/configs/protocol/semantic_mapping_v1.yaml" \
  --priority "$EXTRACTOR/configs/protocol/sensitive_endpoint_priority_v1.yaml" \
  --edge-semantics "$EXTRACTOR/configs/protocol/semantic_edge_mapping_v1.yaml" \
  --extractor-root "$EXTRACTOR" \
  --output "$PROJECT_OUTPUT" \
  --expected 799 --checkpoint-every 10 --progress-every 5

cp -f scripts/care-malben-host-v2/feature_space_contract.txt "$PROJECT_OUTPUT/feature_space_contract.txt"
"$PYTHON_BIN" "$PACKAGE/audit_source_assets_v2.py" \
  --lineage "$PROJECT_OUTPUT/source_lineage_2017.csv" \
  --root "$PROJECT_OUTPUT" --expected 799

cat "$PROJECT_OUTPUT/extraction_summary.txt"
cat "$PROJECT_OUTPUT/source_asset_gate.txt"
echo "Extraction only completed. No OOF model training was started."
