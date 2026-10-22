#!/usr/bin/env bash
# Orchestrates this read-only audit; writes only in its new output directory.
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

ROOT=/home/boyuanjie/projects/tasa-droid
SCRIPTS="$ROOT/scripts/care-background-prototype-audit-v1"
OUT="$ROOT/outputs/care_background_prototype_audit_v1"
PY=/home/boyuanjie/envs/tif/bin/python

MODE=all
RESUME=0
DEVICE=cpu
WORKERS=1
while (($#)); do
  case "$1" in
    --audit-only|--smoke-test|--prototype-support|--cross-bank-overlap|--error-association|--run-controls|--collect) MODE=${1#--}; shift ;;
    --resume) RESUME=1; shift ;;
    --device) DEVICE=${2:?missing device}; shift 2 ;;
    --workers) WORKERS=${2:?missing worker count}; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ "$DEVICE" != cpu ]]; then
  echo 'This audit version supports CPU only; no model training or GPU allocation.' >&2
  exit 2
fi
if [[ "$WORKERS" != 1 ]]; then
  echo 'This audit version is fixed to one low-priority CPU worker.' >&2
  exit 2
fi
cd "$ROOT"
if [[ ! -f "$OUT/preflight.json" ]]; then
  if [[ -e "$OUT" ]]; then
    echo 'Existing nonempty/partial output cannot be adopted without a completed preflight.' >&2
    exit 2
  fi
  "$PY" "$SCRIPTS/audit_assets.py" --audit-only
fi
"$PY" "$SCRIPTS/audit_assets.py" --verify-readonly

run_support() {
  local cache_count=0
  if [[ -d "$OUT/bank_cache" ]]; then
    cache_count=$(find "$OUT/bank_cache" -maxdepth 1 -type f -name '*.npz' | wc -l)
  fi
  if [[ "$cache_count" == 12 ]]; then
    if ((RESUME)); then return; fi
    echo 'Support phase already exists; use --resume to skip completed work.' >&2
    exit 2
  fi
  if [[ "$cache_count" != 0 && "$RESUME" == 0 ]]; then
    echo 'Partial support cache exists; use --resume to continue safely.' >&2
    exit 2
  fi
  "$PY" "$SCRIPTS/compute_prototype_support.py" --prototype-support --resume --device "$DEVICE" --workers "$WORKERS"
}
run_analysis() {
  if [[ -e "$OUT/analysis_phase.json" ]]; then
    if ((RESUME)); then return; fi
    echo 'Analysis phase already exists; use --resume to skip completed work.' >&2
    exit 2
  fi
  "$PY" "$SCRIPTS/analyze.py" --collect --resume --device "$DEVICE" --workers "$WORKERS"
}
run_errors() {
  if [[ -e "$OUT/error_phase.json" ]]; then
    if ((RESUME)); then return; fi
    echo 'Error phase already exists; use --resume to skip completed work.' >&2
    exit 2
  fi
  "$PY" "$SCRIPTS/analyze_errors.py" --error-association
}
run_collect() {
  if [[ -e "$OUT/final_decision.json" ]]; then
    if ((RESUME)); then
      "$PY" "$SCRIPTS/audit_assets.py" --verify-readonly
      return
    fi
    echo 'Final result already exists; use --resume to verify without overwriting.' >&2
    exit 2
  fi
  "$PY" "$SCRIPTS/collect_results.py" --collect
}

case "$MODE" in
  audit-only) ;;
  smoke-test) "$PY" "$SCRIPTS/compute_prototype_support.py" --smoke-test --device "$DEVICE" --workers "$WORKERS" ;;
  prototype-support) run_support ;;
  cross-bank-overlap|run-controls) run_analysis ;;
  error-association) run_errors ;;
  collect) run_collect ;;
  all)
    "$PY" "$SCRIPTS/compute_prototype_support.py" --smoke-test --device "$DEVICE" --workers "$WORKERS"
    run_support
    run_analysis
    run_errors
    run_collect
    ;;
esac
