#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${CARE_ML_PYTHON:-/home/boyuanjie/envs/tif/bin/python}";ROOT="artifacts/care-di25-q-drop-role-source-v1";SCRIPT="scripts/care-di25-q-drop-role-source-v1";mkdir -p "$ROOT"
"$PY" "$SCRIPT/selftest.py";"$PY" "$SCRIPT/preflight.py"
complete=0
if "$PY" -c "import csv,pathlib;p=pathlib.Path('$ROOT/arm/source_predictions.csv');r=list(csv.DictReader(p.open(encoding='utf-8-sig'))) if p.is_file() else [];assert len(r)==799" >/dev/null 2>&1;then complete=1;fi
if [[ $complete -eq 0 ]];then
  if [[ -n "${CARE_GPU:-}" ]];then GPU="$CARE_GPU";else GPU="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2,2n | head -n1 | cut -d, -f1 | tr -d ' ')";fi
  echo "[run] Q_DROP_ROLE_DIRECT physical_GPU=$GPU"
  CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 5 "$PY" "$SCRIPT/run_arm.py" --device cuda 2>&1|tee "$ROOT/arm.log"
else echo "[resume] Q_DROP_ROLE_DIRECT already complete";fi
"$PY" "$SCRIPT/summarize.py";echo "Q_DROP_ROLE_SOURCE_EXPERIMENT_DONE"
