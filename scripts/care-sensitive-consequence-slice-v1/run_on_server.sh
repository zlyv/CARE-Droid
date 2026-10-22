#!/usr/bin/env bash
set -uo pipefail

PROJECT="${SCS_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${SCS_OUTPUT:-$PROJECT/outputs/care-sensitive-consequence-slice-v1}"
PY="${SCS_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
export SCS_PROJECT="$PROJECT" SCS_OUTPUT="$OUT"
mkdir -p "$OUT/logs" "$OUT/cache" "$OUT/checkpoints"
exec > >(tee -a "$OUT/logs/run.log") 2>&1

echo "CARE_SCS_PIPELINE_START project=$PROJECT output=$OUT python=$PY"
if [[ ! -x "$PY" ]]; then echo "BLOCKED_BY_CONTRACT: python missing $PY"; exit 2; fi

phase() {
  local name="$1"; shift
  echo "SCS_PHASE_START name=$name epoch=$(date +%s)"
  "$@"; local rc=$?
  echo "SCS_PHASE_END name=$name rc=$rc epoch=$(date +%s)"
  return $rc
}

phase syntax "$PY" -m py_compile "$SCRIPT_DIR"/*.py || exit $?
phase selftest "$PY" "$SCRIPT_DIR/selftest.py" || exit $?
phase preflight "$PY" "$SCRIPT_DIR/preflight.py" || exit $?
phase features "$PY" "$SCRIPT_DIR/extract_features.py" || exit $?
phase coverage "$PY" "$SCRIPT_DIR/coverage_gate.py"
RC=$?
if [[ $RC -eq 20 ]]; then
  phase integrity "$PY" "$SCRIPT_DIR/finalize.py" || exit $?
  phase summary "$PY" "$SCRIPT_DIR/summarize.py" || exit $?
  echo "CARE_SCS_PIPELINE_STOPPED_AT=LABEL_FREE_SLICE_GATE"
  exit 0
elif [[ $RC -ne 0 ]]; then
  exit $RC
fi
phase direction "$PY" "$SCRIPT_DIR/direction_probe.py" || exit $?
phase integrity "$PY" "$SCRIPT_DIR/finalize.py" || exit $?
phase summary "$PY" "$SCRIPT_DIR/summarize.py" || exit $?
echo "CARE_SENSITIVE_CONSEQUENCE_SLICE_SOURCE_PIPELINE_DONE"
echo "output=$OUT"
echo "report=$OUT/report.md"
echo "decision=$OUT/final_decision.json"
