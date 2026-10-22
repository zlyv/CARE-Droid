#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/boyuanjie/projects/tasa-droid"
cd "$PROJECT"
PACKAGE="scripts/care-source-recovery-stage2only-v1"
MALBEN="scripts/care-malben-contrast-v2"
HDD_BASE="${CARE_RECOVERY_ROOT:-/mnt/HDD1/boyuanjie/tasa-droid-storage/care-recovery}"
PREFLIGHT="artifacts/care-source-recovery-stage2only-v1/preflight"
mkdir -p "$PREFLIGHT" "$HDD_BASE"

exec 9>"$PREFLIGHT/.preflight.lock"
if ! flock -n 9; then
  echo "Another preflight process owns $PREFLIGHT/.preflight.lock" >&2
  exit 1
fi
if pgrep -af 'care-source-recovery-stage2only-v1/run_full_source_on_server.sh' >/dev/null 2>&1; then
  echo "The old full source runner is still active. Stop it before preflight repair:" >&2
  pgrep -af 'care-source-recovery-stage2only-v1/run_full_source_on_server.sh' >&2 || true
  exit 1
fi
if pgrep -af 'run_compact_feature_extraction_ag.py' >/dev/null 2>&1; then
  echo "A compact feature extractor is still running. Stop duplicate writers before integrity repair:" >&2
  pgrep -af 'run_compact_feature_extraction_ag.py' >&2 || true
  exit 1
fi
if pgrep -af 'run_endpoint_centered_tensor_cache_ag.py' >/dev/null 2>&1; then
  echo "An endpoint-cache extractor is already running. Stop the old full job before the bounded Ego smoke test:" >&2
  pgrep -af 'run_endpoint_centered_tensor_cache_ag.py' >&2 || true
  exit 1
fi

ML_PY="${PYTHON_BIN:-/home/boyuanjie/envs/tif/bin/python}"
AG_PY="${ANDRO_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
"$ML_PY" -c "import numpy, sklearn, torch; print('ML deps:', numpy.__version__, sklearn.__version__, torch.__version__, 'cuda=' + str(torch.cuda.is_available()))"
"$AG_PY" -c "import androguard, yaml, networkx; print('Androguard dependency check passed')"

"$ML_PY" "$PACKAGE/selftest_recovery.py"
"$ML_PY" "$PACKAGE/selftest_integrity_audit.py"
"$ML_PY" "$MALBEN/selftest_stage2only.py"
"$ML_PY" "$MALBEN/selftest_role_prototypes.py"
"$ML_PY" "$MALBEN/selftest_model_contract.py"
"$ML_PY" "$PACKAGE/prepare_recovery.py"

LINEAGE="artifacts/care-source-recovery-stage2only-v1/source_lineage_2017.csv"
FEATURES="artifacts/p1-features/features.csv"
FAILURES="artifacts/p1-features/feature_failures.csv"
if [[ ! -f "$FEATURES" ]]; then
  echo "Missing compact source features: $FEATURES" >&2
  echo "This preflight does not start a new 799-APK extraction." >&2
  exit 1
fi
"$ML_PY" "$PACKAGE/audit_repair_source_features.py" \
  --lineage "$LINEAGE" \
  --features "$FEATURES" \
  --failures "$FAILURES" \
  --output "$PREFLIGHT" \
  --expected 799 \
  --repair-identical-duplicates

EXTRACTOR="$PACKAGE/extractor_repo"
MAPPING="$EXTRACTOR/configs/protocol/semantic_mapping_v1.yaml"
PRIORITY="$EXTRACTOR/configs/protocol/sensitive_endpoint_priority_v1.yaml"
EDGE="$EXTRACTOR/configs/protocol/semantic_edge_mapping_v1.yaml"
export PYTHONPATH="$PACKAGE:$EXTRACTOR/src:$EXTRACTOR/scripts${PYTHONPATH:+:$PYTHONPATH}"

PROBE_LINEAGE="$PREFLIGHT/source_lineage_fixed50_probe.csv"
PROBE_CACHE_DIR="$HDD_BASE/ego-smoke-endpoint-cache-fixed50"
PROBE_CACHE="$PROBE_CACHE_DIR/endpoint_centered_tensor_cache.jsonl.gz"
mkdir -p "$PROBE_CACHE_DIR"
"$ML_PY" "$PACKAGE/make_smoke_lineage.py" \
  --input "$LINEAGE" --output "$PROBE_LINEAGE" --count 50
if "$ML_PY" "$PACKAGE/check_complete.py" jsonl_gz "$PROBE_CACHE" --expected 50; then
  echo "Fixed-50 endpoint probe cache already complete."
else
  "$AG_PY" "$EXTRACTOR/scripts/run_endpoint_centered_tensor_cache_ag.py" \
    --lineage "$PROBE_LINEAGE" \
    --mapping "$MAPPING" \
    --priority "$PRIORITY" \
    --edge-semantics "$EDGE" \
    --output "$PROBE_CACHE_DIR" \
    --limit 0 \
    --resume \
    --endpoint-cap 256 \
    --progress-every 5 \
    --min-free-gb 2
fi
"$ML_PY" "$PACKAGE/check_complete.py" jsonl_gz "$PROBE_CACHE" --expected 50
"$ML_PY" "$PACKAGE/audit_true_ego_smoke.py" \
  --cache "$PROBE_CACHE" \
  --mapping "$MAPPING" \
  --output "$PREFLIGHT" \
  --min-apks 10 \
  --target-apks 10

echo
echo "Both automatic preflight gates passed. No OOF training or 799-Ego extraction was started."
echo "Inspect before approval:"
echo "  cat $PREFLIGHT/source_lineage_integrity_audit.txt"
echo "  cat $PREFLIGHT/ego_smoke_test_summary.txt"
echo "  column -s, -t < $PREFLIGHT/source_duplicate_rows.csv | less -S"
echo "  column -s, -t < $PREFLIGHT/ego_smoke_test_examples.csv | less -S"
echo "Do not approve until these reports have been reviewed."
