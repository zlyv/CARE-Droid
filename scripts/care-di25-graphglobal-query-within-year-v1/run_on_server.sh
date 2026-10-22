#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid

PACKAGE="scripts/care-di25-graphglobal-query-within-year-v1"
ROOT="artifacts/care-di25-graphglobal-query-within-year-v1"
PYTHON="${ML_PYTHON_BIN:-${PYTHON_BIN:-/home/boyuanjie/envs/tif/bin/python}}"
mkdir -p "$ROOT/arms" "$ROOT/logs" "$ROOT/results"
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

if [[ ! -x "$PYTHON" ]] || ! "$PYTHON" -c 'import numpy,sklearn,torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
  echo "BLOCKED_BY_CONTRACT: CUDA CARE Python unavailable: $PYTHON" >&2
  exit 1
fi
"$PYTHON" "$PACKAGE/selftest.py"
"$PYTHON" "$PACKAGE/preflight.py"

run_arm() {
  local arm="$1"
  local gpu="$2"
  local output="$ROOT/arms/$arm"
  local prediction="$output/within_year_predictions.csv"
  if [[ -s "$prediction" ]] && [[ "$(($(wc -l < "$prediction")-1))" -eq 1989 ]]; then
    echo "[resume] $arm already has 1989 OOF predictions"
    return 0
  fi
  mkdir -p "$output"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$PACKAGE/run_within_year_arm.py" \
    --arm "$arm" --device cuda --seeds 900000,900001,900002 --max-delta 2.25 \
    --output "$output" 2>&1 | tee "$ROOT/logs/$arm.log"
}

run_arm N0_NO_GRAPHGLOBAL 0 & pid0=$!
run_arm Q_QUERY_ONLY 1 & pid1=$!
status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
if [[ "$status" -ne 0 ]]; then
  echo "BLOCKED_BY_CONTRACT: at least one same-year arm failed; inspect $ROOT/logs" >&2
  exit 1
fi
"$PYTHON" "$PACKAGE/summarize.py"
cat "$ROOT/results/REPORT.txt"
echo "GRAPHGLOBAL_QUERY_WITHIN_YEAR_DONE"
