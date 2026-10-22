#!/usr/bin/env bash
set -uo pipefail
PROJECT="${FPR_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${FPR_OUTPUT:-$PROJECT/outputs/care-fp-count-matched-residual-v1}"
PY="${FPR_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
export FPR_PROJECT="$PROJECT" FPR_OUTPUT="$OUT"
mkdir -p "$OUT/logs"
exec > >(tee -a "$OUT/logs/run.log") 2>&1

phase() { local name="$1"; shift; echo "FPR_PHASE_START name=$name epoch=$(date +%s)"; "$@"; local rc=$?; echo "FPR_PHASE_END name=$name rc=$rc epoch=$(date +%s)"; return $rc; }

echo "CARE_FPR_PIPELINE_START project=$PROJECT output=$OUT python=$PY"
if [[ ! -x "$PY" ]]; then echo "BLOCKED_BY_CONTRACT: python missing $PY"; exit 2; fi
phase syntax "$PY" -m py_compile "$SCRIPT_DIR"/*.py || exit $?
phase selftest "$PY" "$SCRIPT_DIR/selftest.py" || exit $?
phase preflight "$PY" "$SCRIPT_DIR/preflight.py" || exit $?
phase audit "$PY" "$SCRIPT_DIR/run_audit.py" || exit $?
phase integrity "$PY" "$SCRIPT_DIR/finalize.py" || exit $?
phase summary "$PY" "$SCRIPT_DIR/summarize.py" || exit $?
echo "CARE_FP_COUNT_MATCHED_RESIDUAL_PIPELINE_DONE"
echo "output=$OUT"
echo "report=$OUT/report.md"
echo "decision=$OUT/final_decision.json"
