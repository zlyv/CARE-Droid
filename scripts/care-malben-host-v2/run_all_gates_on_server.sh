#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PYTHON_BIN="${PYTHON_BIN:-/home/boyuanjie/envs/tif/bin/python}"
ASSETS="artifacts/care-source-local-assets-v2"

"$PYTHON_BIN" scripts/care-source-local-assets-v2/audit_source_lineage_v2.py \
  --lineage "$ASSETS/source_lineage_2017.csv" \
  --folds artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv \
  --features artifacts/p1-features/features.csv \
  --output "$ASSETS/source_lineage_gate.json"
"$PYTHON_BIN" scripts/care-source-local-assets-v2/selftest_exclusive_writer.py
"$PYTHON_BIN" scripts/care-source-local-assets-v2/selftest_asset_reconstruction.py
"$PYTHON_BIN" scripts/care-source-local-assets-v2/audit_source_assets_v2.py \
  --lineage "$ASSETS/source_lineage_2017.csv" --root "$ASSETS" --expected 799
"$PYTHON_BIN" scripts/care-malben-host-v2/audit_source_label_contract.py \
  --features artifacts/p1-features/features.csv \
  --folds artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv \
  --output "$ASSETS/source_label_contract_gate.json"
test -s scripts/care-malben-host-v2/feature_space_contract.txt
grep -q '74-D' scripts/care-malben-host-v2/feature_space_contract.txt
"$PYTHON_BIN" scripts/care-malben-host-v2/selftest_shared_role_prototypes.py
"$PYTHON_BIN" scripts/care-stage2only-malben-v1/selftest_stage2only_contract.py
"$PYTHON_BIN" scripts/care-malben-host-v2/selftest_selective_parameter_contract.py
echo "ALL_9_PREDECLARED_GATES_PASS"
