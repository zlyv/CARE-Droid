#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
export PYTHONDONTWRITEBYTECODE=1
export CLD_OUTPUT=${CLD_OUTPUT:-/home/boyuanjie/projects/tasa-droid/outputs/care_crossyear_layerwise_diagnosis_v1}
mkdir -p "$CLD_OUTPUT"/tmp "$CLD_OUTPUT"/cache
export TMPDIR="$CLD_OUTPUT/tmp"
export XDG_CACHE_HOME="$CLD_OUTPUT/cache"
export MPLCONFIGDIR="$CLD_OUTPUT/cache/matplotlib"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
HERE=/home/boyuanjie/projects/tasa-droid/scripts/care-crossyear-layerwise-diagnosis-v1
PY=/home/boyuanjie/envs/tif/bin/python
case "${1:-}" in
  --audit-only) "$PY" "$HERE/selftest.py"; "$PY" "$HERE/audit_replay.py";;
  --extract-layers) "$PY" "$HERE/extract_layers.py";;
  --run-probes|--diagnose) "$PY" "$HERE/probes.py";;
  --collect) "$PY" "$HERE/collect.py";;
  *) echo 'Usage: run.sh --audit-only|--extract-layers|--run-probes|--collect' >&2; exit 2;;
esac
