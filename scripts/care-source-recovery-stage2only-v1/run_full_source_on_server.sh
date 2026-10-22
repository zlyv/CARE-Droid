#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/boyuanjie/projects/tasa-droid"
cd "$PROJECT"

PACKAGE="scripts/care-source-recovery-stage2only-v1"
MALBEN="scripts/care-malben-contrast-v2"
HDD_BASE="${CARE_RECOVERY_ROOT:-/mnt/HDD1/boyuanjie/tasa-droid-storage/care-recovery}"
mkdir -p "$HDD_BASE" artifacts

# A full recovery process is a single-writer pipeline.  This lock prevents two
# tmux sessions from computing the same stale resume state and appending the
# same APK rows concurrently.
LOCK_ROOT="artifacts/care-source-recovery-stage2only-v1"
mkdir -p "$LOCK_ROOT"
exec 8>"$LOCK_ROOT/.full_pipeline.lock"
if ! flock -n 8; then
  echo "Another full source-recovery pipeline owns $LOCK_ROOT/.full_pipeline.lock" >&2
  exit 1
fi

choose_ml_python() {
  local candidates=()
  if [[ -n "${PYTHON_BIN:-}" ]]; then candidates+=("$PYTHON_BIN"); fi
  candidates+=(
    /home/boyuanjie/envs/tif/bin/python
    /home/boyuanjie/venvs/malscan/bin/python
    /home/boyuanjie/envs/malscan/bin/python
  )
  if command -v python3 >/dev/null 2>&1; then candidates+=("$(command -v python3)"); fi
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]] && "$candidate" -c "import numpy, sklearn, torch" >/dev/null 2>&1; then
      echo "$candidate"; return 0
    fi
  done
  return 1
}

choose_ag_python() {
  local candidates=()
  if [[ -n "${ANDRO_PYTHON:-}" ]]; then candidates+=("$ANDRO_PYTHON"); fi
  candidates+=(
    /home/boyuanjie/envs/androguard/bin/python
    /home/boyuanjie/venvs/androguard/bin/python
    /home/boyuanjie/envs/tif/bin/python
  )
  if command -v python3 >/dev/null 2>&1; then candidates+=("$(command -v python3)"); fi
  while IFS= read -r candidate; do candidates+=("$candidate"); done < <(
    find /home/boyuanjie -maxdepth 4 \( -type f -o -type l \) -path '*/bin/python' 2>/dev/null || true
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]] && "$candidate" -c "import androguard, yaml, networkx" >/dev/null 2>&1; then
      echo "$candidate"; return 0
    fi
  done
  return 1
}

ML_PY="$(choose_ml_python || true)"
AG_PY="$(choose_ag_python || true)"
if [[ -z "$ML_PY" ]]; then
  echo "No Python environment with numpy/sklearn/torch. Set PYTHON_BIN." >&2
  exit 1
fi
if [[ -z "$AG_PY" ]]; then
  echo "No Python environment with androguard/yaml/networkx. Set ANDRO_PYTHON." >&2
  exit 1
fi
echo "ML Python: $ML_PY"
echo "Androguard Python: $AG_PY"
"$ML_PY" -c "import numpy, sklearn, torch; print('ML deps:', numpy.__version__, sklearn.__version__, torch.__version__, 'cuda=' + str(torch.cuda.is_available()))"
"$AG_PY" -c "import androguard, yaml, networkx; print('Androguard dependency check passed')"

ensure_hdd_link() {
  local project_path="$1"
  local hdd_path="$2"
  mkdir -p "$hdd_path"
  if [[ -L "$project_path" ]]; then
    local resolved
    resolved="$(readlink -f "$project_path")"
    if [[ "$resolved" != "$(readlink -f "$hdd_path")" ]]; then
      echo "Existing symlink points elsewhere: $project_path -> $resolved" >&2
      exit 1
    fi
    return 0
  fi
  if [[ -e "$project_path" ]]; then
    if [[ -d "$project_path" ]] && [[ -z "$(find "$project_path" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      rmdir "$project_path"
    else
      echo "Refusing to replace non-empty project path: $project_path" >&2
      echo "Move it manually, then rerun. No existing data was changed." >&2
      exit 1
    fi
  fi
  mkdir -p "$(dirname "$project_path")"
  ln -s "$hdd_path" "$project_path"
  echo "HDD-backed cache: $project_path -> $hdd_path"
}

# Only heavy generated caches go to HDD1.  Scripts and compact reports remain
# in the project tree.
ensure_hdd_link \
  "artifacts/p25-endpoint-cache-full-cap256" \
  "$HDD_BASE/p25-endpoint-cache-full-cap256"
ensure_hdd_link \
  "artifacts/care-local-region-cache-v1" \
  "$HDD_BASE/care-local-region-cache-v1"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-8}"

"$ML_PY" "$PACKAGE/selftest_recovery.py"
"$ML_PY" "$MALBEN/selftest_stage2only.py"
"$ML_PY" "$MALBEN/selftest_role_prototypes.py"
"$ML_PY" "$MALBEN/selftest_model_contract.py"
"$ML_PY" "$PACKAGE/prepare_recovery.py"

LINEAGE="artifacts/care-source-recovery-stage2only-v1/source_lineage_2017.csv"
EXTRACTOR="$PACKAGE/extractor_repo"
MAPPING="$EXTRACTOR/configs/protocol/semantic_mapping_v1.yaml"
PRIORITY="$EXTRACTOR/configs/protocol/sensitive_endpoint_priority_v1.yaml"
EDGE="$EXTRACTOR/configs/protocol/semantic_edge_mapping_v1.yaml"
export PYTHONPATH="$EXTRACTOR/src:$EXTRACTOR/scripts${PYTHONPATH:+:$PYTHONPATH}"

# The preflight is a separate, short program.  This full runner never repairs
# source features or creates smoke assets.  It only continues when both exact
# gate files have passed and their combined digest was manually approved.
PREFLIGHT="artifacts/care-source-recovery-stage2only-v1/preflight"
"$ML_PY" "$PACKAGE/preflight_approval.py" check --root "$PREFLIGHT"
"$ML_PY" "$PACKAGE/check_complete.py" feature_csv artifacts/p1-features/features.csv
echo "Source-lineage integrity and true-Ego smoke gates are approved."

if "$ML_PY" "$PACKAGE/check_complete.py" feature_csv artifacts/p1-features/features.csv; then
  echo "Compact source features already complete; skipping extraction."
else
  "$AG_PY" "$EXTRACTOR/scripts/run_compact_feature_extraction_ag.py" \
    --lineage "$LINEAGE" \
    --mapping "$MAPPING" \
    --priority "$PRIORITY" \
    --output artifacts/p1-features \
    --progress-every 20 \
    --min-free-gb 2
fi
"$ML_PY" "$PACKAGE/check_complete.py" feature_csv artifacts/p1-features/features.csv

CACHE="artifacts/p25-endpoint-cache-full-cap256/endpoint_centered_tensor_cache.jsonl.gz"
if "$ML_PY" "$PACKAGE/check_complete.py" jsonl_gz "$CACHE"; then
  echo "Endpoint graph cache already complete; skipping extraction."
else
  "$AG_PY" "$EXTRACTOR/scripts/run_endpoint_centered_tensor_cache_ag.py" \
    --lineage "$LINEAGE" \
    --mapping "$MAPPING" \
    --priority "$PRIORITY" \
    --edge-semantics "$EDGE" \
    --output artifacts/p25-endpoint-cache-full-cap256 \
    --limit 0 \
    --resume \
    --endpoint-cap 256 \
    --progress-every 10 \
    --min-free-gb 2
fi
"$ML_PY" "$PACKAGE/check_complete.py" jsonl_gz "$CACHE"

BASELINE="artifacts/care-di-budget-bound-selection-v1/retrained_source/max_delta_2p25/source_predictions.csv"
if "$ML_PY" "$PACKAGE/check_complete.py" csv "$BASELINE"; then
  echo "Selected anomaly baseline already complete; skipping retraining."
else
  "$ML_PY" scripts/care-rgcondattn-dedup-bounded-source-v1/run_care_rgcondattn_dedup_bounded_source_v1.py \
    --variants RGCondAttn \
    --seeds 0,1,2 \
    --max-delta 2.25 \
    --device auto \
    --output artifacts/care-di-budget-bound-selection-v1/retrained_source/max_delta_2p25
fi
"$ML_PY" "$PACKAGE/check_complete.py" csv "$BASELINE"

MALBEN_PRED="artifacts/care-malben-contrast-v2/source/model/source_predictions.csv"
if "$ML_PY" "$PACKAGE/check_complete.py" csv "$MALBEN_PRED"; then
  echo "MalBen OOF models already complete; skipping retraining."
else
  "$ML_PY" "$MALBEN/run_care_malben_role_source_v2.py" \
    --variants BenignOnly,MaliciousOnly,MalBenContrast \
    --seeds 0,1,2 \
    --max-delta 2.25 \
    --device auto \
    --output artifacts/care-malben-contrast-v2/source/model
fi
"$ML_PY" "$PACKAGE/check_complete.py" csv "$MALBEN_PRED"

STAGE2_PRED="artifacts/care-malben-contrast-v2/source/stage2only/source_predictions.csv"
if "$ML_PY" "$PACKAGE/check_complete.py" csv "$STAGE2_PRED"; then
  echo "StageIIOnly-MalBen OOF result already complete; skipping retraining."
else
  "$ML_PY" "$MALBEN/run_stage2only_malben_source_v1.py" \
    --seeds 0,1,2 \
    --device auto \
    --output artifacts/care-malben-contrast-v2/source/stage2only
fi
"$ML_PY" "$PACKAGE/check_complete.py" csv "$STAGE2_PRED"

"$ML_PY" "$MALBEN/summarize_stage2only_main_table.py"
cat artifacts/care-malben-contrast-v2/source/main_comparison/REPORT.txt

echo "Source-only recovery and Stage-II-only experiment completed."
echo "Heavy endpoint/region caches remain under: $HDD_BASE"
