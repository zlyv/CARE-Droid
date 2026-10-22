#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/boyuanjie/projects/tasa-droid"
cd "$PROJECT"
export PYTHONNOUSERSITE=1
PKG="scripts/care-recovery-missing-assets-v1"
RECOVERY="scripts/care-source-recovery-stage2only-v1"
EXTRACTOR="$RECOVERY/extractor_repo"

choose_python() {
  local imports="$1"
  shift
  local candidate
  local -a candidates=("$@")
  while IFS= read -r candidate; do candidates+=("$candidate"); done < <(
    find /home/boyuanjie -maxdepth 5 \( -type f -o -type l \) -path '*/bin/python' 2>/dev/null | sort -u
  )
  if command -v python3 >/dev/null 2>&1; then candidates+=("$(command -v python3)"); fi
  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]] && PYTHONNOUSERSITE=1 "$candidate" -c "$imports" >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

ML_PY="$(choose_python \
  'import numpy, sklearn, torch, androguard, yaml, networkx' \
  /home/boyuanjie/envs/tif/bin/python \
  /home/boyuanjie/venvs/tif/bin/python \
  /home/boyuanjie/envs/androguard/bin/python \
  /home/boyuanjie/venvs/androguard/bin/python || true)"
ROUTER_PY="$(choose_python \
  'import numpy, sklearn, torch, xgboost' \
  /home/boyuanjie/venvs/care-recovery312/bin/python \
  /home/boyuanjie/venvs/malscan/bin/python \
  /home/boyuanjie/envs/malscan/bin/python \
  /home/boyuanjie/envs/tif/bin/python \
  /home/boyuanjie/venvs/tif/bin/python || true)"

if [[ -z "$ML_PY" ]]; then
  echo "BLOCKED_BY_CONTRACT: no existing Python has numpy/sklearn/torch/androguard/yaml/networkx" >&2
  echo "Candidates audited under /home/boyuanjie/*/*/bin/python; no package was installed." >&2
  exit 1
fi
if [[ -z "$ROUTER_PY" ]]; then
  echo "BLOCKED_BY_CONTRACT: no existing Python has numpy/sklearn/torch/xgboost" >&2
  echo "Candidates audited under /home/boyuanjie/*/*/bin/python; no package was installed." >&2
  exit 1
fi
echo "ML Python: $ML_PY"
echo "Router Python: $ROUTER_PY"
PYTHONNOUSERSITE=1 "$ML_PY" -c "import numpy, sklearn, torch, androguard; print('ML deps:', numpy.__version__, sklearn.__version__, torch.__version__)"
PYTHONNOUSERSITE=1 "$ROUTER_PY" -c "import numpy, sklearn, torch, xgboost; print('Router deps:', numpy.__version__, sklearn.__version__, torch.__version__, xgboost.__version__)"

# Restore the immutable assets bundled in the already-audited source recovery.
"$ML_PY" "$RECOVERY/selftest_recovery.py"
"$ML_PY" "$RECOVERY/prepare_recovery.py"

LINEAGE="artifacts/care-source-recovery-stage2only-v1/source_lineage_2017.csv"
MAPPING="$EXTRACTOR/configs/protocol/semantic_mapping_v1.yaml"
PRIORITY="$EXTRACTOR/configs/protocol/sensitive_endpoint_priority_v1.yaml"
FEATURES="artifacts/p1-features/features.csv"

# Compact76 cannot be reconstructed exactly from the bounded endpoint cache.
# This is the only APK analysis in this recovery job and is append-resumable.
if "$ML_PY" "$RECOVERY/check_complete.py" feature_csv "$FEATURES"; then
  echo "Compact source features already complete; skipping APK extraction."
else
  export PYTHONPATH="$EXTRACTOR/src:$EXTRACTOR/scripts${PYTHONPATH:+:$PYTHONPATH}"
  "$ML_PY" "$EXTRACTOR/scripts/run_compact_feature_extraction_ag.py" \
    --lineage "$LINEAGE" \
    --mapping "$MAPPING" \
    --priority "$PRIORITY" \
    --output artifacts/p1-features \
    --progress-every 20 \
    --min-free-gb 2
fi
"$ML_PY" "$RECOVERY/check_complete.py" feature_csv "$FEATURES"

# Refit only the source OOF committee. The frozen Stage-I/expert probabilities
# come from the immutable bundled 799-row reference; Stage-II is not retrained.
"$ROUTER_PY" scripts/care-committee-uncertainty-router-v1/selftest_committee_uncertainty.py
"$ROUTER_PY" scripts/care-committee-uncertainty-router-v1/run_committee_source_v1.py \
  --features "$FEATURES" \
  --schema artifacts/p1-features/feature_schema.json \
  --p72-oof artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv \
  --expert-predictions artifacts/global24-a25-route-conditioned-joints7-v1-full/source_predictions.csv \
  --output artifacts/care-committee-uncertainty-router-source-v1

# Recover DI25 by its frozen formula and exact per-fold 25% selection. No
# future score file or future label is read by this source-only builder.
"$ROUTER_PY" "$PKG/build_source_di_only.py" \
  --input artifacts/care-committee-uncertainty-router-source-v1/source_sample_scores.csv \
  --output artifacts/care-committee-decision-instability-router-source-v1/source_sample_scores.csv \
  --mirror artifacts/care-route-stage2-future-matrix-v1/preflight/source_sample_scores.csv

"$ML_PY" "$PKG/verify_recovered_assets.py"
if [[ -f care-droid-code-recovery-v1/check_required_assets.py ]]; then
  "$ML_PY" care-droid-code-recovery-v1/check_required_assets.py
elif [[ -f scripts/care-droid-code-recovery-v1/check_required_assets.py ]]; then
  "$ML_PY" scripts/care-droid-code-recovery-v1/check_required_assets.py
else
  echo "Recovery inventory checker not found; the stricter four-asset gate above passed."
fi
echo "CARE_DROID_MISSING_ASSETS_RECOVERY_DONE"
