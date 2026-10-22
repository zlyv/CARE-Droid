#!/usr/bin/env bash
set -uo pipefail

PROJECT="${CMW_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${CMW_OUTPUT:-$PROJECT/outputs/care-contextual-multihop-witness-v1}"
PY="${CMW_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
export CMW_PROJECT="$PROJECT" CMW_OUTPUT="$OUT"
mkdir -p "$OUT/logs" "$OUT/checkpoints" "$OUT/cache"
exec > >(tee -a "$OUT/logs/run.log") 2>&1

echo "CARE_CMW_PIPELINE_START project=$PROJECT output=$OUT python=$PY"
if [[ ! -x "$PY" ]]; then echo "BLOCKED_BY_CONTRACT: python missing $PY"; exit 2; fi

phase() {
  local name="$1"; shift
  echo "CMW_PHASE_START name=$name epoch=$(date +%s)"
  "$@"
  local rc=$?
  echo "CMW_PHASE_END name=$name rc=$rc epoch=$(date +%s)"
  return $rc
}

phase syntax "$PY" -m py_compile "$SCRIPT_DIR"/*.py || exit $?
phase selftest "$PY" "$SCRIPT_DIR/selftest.py" || exit $?
phase preflight "$PY" "$SCRIPT_DIR/preflight.py" || exit $?
phase source_features "$PY" "$SCRIPT_DIR/prepare_features.py" --cohort source --workers "${CMW_GRAPH_WORKERS:-2}" || exit $?

phase coverage_gate "$PY" "$SCRIPT_DIR/coverage_gate.py"
RC=$?
if [[ $RC -eq 20 ]]; then
  phase integrity "$PY" "$SCRIPT_DIR/finalize.py"
  phase summary "$PY" "$SCRIPT_DIR/summarize.py"
  echo "CARE_CMW_PIPELINE_STOPPED_AT=LABEL_FREE_COVERAGE_GATE"
  exit 0
elif [[ $RC -ne 0 ]]; then
  exit $RC
fi

phase source_probe "$PY" "$SCRIPT_DIR/run_source.py"
RC=$?
if [[ $RC -eq 21 ]]; then
  echo "CMW_SOURCE_FROZEN diagnostics_authorized_by_user=true source_supported=false future_cannot_change_source_decision=true"
elif [[ $RC -ne 0 ]]; then
  exit $RC
else
  echo "CMW_SOURCE_FROZEN diagnostics_authorized_by_user=true source_supported=true"
fi

phase future_frozen_features "$PY" "$SCRIPT_DIR/prepare_features.py" --cohort future_frozen --workers "${CMW_GRAPH_WORKERS:-2}" || exit $?
phase future_within_features "$PY" "$SCRIPT_DIR/prepare_features.py" --cohort future_within --workers "${CMW_GRAPH_WORKERS:-2}" || exit $?
phase diagnostics "$PY" "$SCRIPT_DIR/run_diagnostics.py" || exit $?
phase integrity "$PY" "$SCRIPT_DIR/finalize.py" || exit $?
phase summary "$PY" "$SCRIPT_DIR/summarize.py" || exit $?

echo "CARE_CONTEXTUAL_MULTIHOP_WITNESS_PIPELINE_DONE"
echo "output=$OUT"
echo "report=$OUT/report.md"
echo "final_decision=$OUT/final_decision.json"
