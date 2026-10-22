#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${TBFRS_OUTPUT:-$ROOT/outputs/care-tb-frs-source-v1}"
PY="${TBFRS_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
mkdir -p "$OUT/logs" "$OUT/checkpoints"
phase(){
  local name="$1"; shift; local mark="$OUT/checkpoints/$name.done"
  if [[ -f "$mark" ]]; then echo "TBFRS_PHASE_RESUME name=$name"; return; fi
  echo "TBFRS_PHASE_START name=$name epoch=$(date +%s)"
  "$@" 2>&1 | tee "$OUT/logs/$name.log"
  touch "$mark"; echo "TBFRS_PHASE_DONE name=$name epoch=$(date +%s)"
}
export TBFRS_PROJECT="$ROOT" TBFRS_OUTPUT="$OUT" PYTHONPATH="$ROOT/scripts/care-tb-frs-source-v1:${PYTHONPATH:-}"
echo "CARE_TBFRS_SOURCE_PIPELINE_START output=$OUT python=$PY"
phase syntax "$PY" -m py_compile "$ROOT"/scripts/care-tb-frs-source-v1/*.py
phase selftest "$PY" "$ROOT/scripts/care-tb-frs-source-v1/selftest.py"
phase preflight "$PY" "$ROOT/scripts/care-tb-frs-source-v1/preflight.py"
phase source_probe "$PY" "$ROOT/scripts/care-tb-frs-source-v1/run_probe.py" --null-repeats 200
phase integrity "$PY" "$ROOT/scripts/care-tb-frs-source-v1/integrity_after.py"
phase summary "$PY" "$ROOT/scripts/care-tb-frs-source-v1/summarize.py"
echo "CARE_TBFRS_SOURCE_PIPELINE_DONE"
echo "decision=$OUT/final_decision.json"
echo "report=$OUT/REPORT.md"
