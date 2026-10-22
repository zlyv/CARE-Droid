#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
export PYTHONDONTWRITEBYTECODE=1
export CLD_OUTPUT=${CLD_OUTPUT:-/home/boyuanjie/projects/tasa-droid/outputs/care_crossyear_layerwise_diagnosis_v2}
mkdir -p "$CLD_OUTPUT"/tmp "$CLD_OUTPUT"/cache
export TMPDIR="$CLD_OUTPUT/tmp"
export XDG_CACHE_HOME="$CLD_OUTPUT/cache"
export MPLCONFIGDIR="$CLD_OUTPUT/cache/matplotlib"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
HERE=/home/boyuanjie/projects/tasa-droid/scripts/care-crossyear-layerwise-diagnosis-v2
PY=/home/boyuanjie/envs/tif/bin/python
PHASE=${1:-}
if [ -z "$PHASE" ]; then echo 'missing phase' >&2; exit 2; fi
shift
RESUME=0
export CLD_WINDOWS=${CLD_WINDOWS:-T2,T3,T4}
export CLD_DEVICE=${CLD_DEVICE:-cpu}
while [ "$#" -gt 0 ]; do
  case "$1" in
    --windows) CLD_WINDOWS=$2; export CLD_WINDOWS; shift 2;;
    --device) CLD_DEVICE=$2; export CLD_DEVICE; shift 2;;
    --resume) RESUME=1; shift;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
case "$CLD_WINDOWS" in
  T2,T3,T4|T2|T3|T4|T2,T3|T2,T4|T3,T4) ;;
  *) echo 'windows must be an ordered subset of T2,T3,T4' >&2; exit 2;;
esac
case "$CLD_DEVICE" in cpu|cuda) ;; *) echo 'device must be cpu or cuda' >&2; exit 2;; esac
if [ "$RESUME" -eq 1 ]; then
  case "$PHASE" in
    --audit-only) MARKER=baseline_replay.json;;
    --extract-layers) MARKER=layer_extraction_audit.json;;
    --run-probes|--diagnose) MARKER=probe_predictions.csv;;
    --collect) MARKER=baseline_integrity_check.json;;
    --smoke-test) MARKER=__none__;;
    *) MARKER=__none__;;
  esac
  if [ -f "$CLD_OUTPUT/$MARKER" ]; then
    echo "CLD_RESUME_COMPLETED phase=$PHASE marker=$MARKER; no overwrite"
    exit 0
  fi
fi
case "$PHASE" in
  --smoke-test) "$PY" "$HERE/selftest.py";;
  --audit-only) "$PY" "$HERE/selftest.py"; "$PY" "$HERE/audit_replay.py";;
  --extract-layers) "$PY" "$HERE/extract_layers.py";;
  --run-probes|--diagnose) "$PY" "$HERE/probes.py";;
  --collect)
    if [ "$CLD_WINDOWS" != T2,T3,T4 ]; then echo 'final collection requires all three windows' >&2; exit 2; fi
    "$PY" "$HERE/collect.py";;
  *) echo 'Usage: run.sh --smoke-test|--audit-only|--extract-layers|--run-probes|--collect [--windows T2,T3,T4] [--device cpu|cuda] [--resume]' >&2; exit 2;;
esac
