#!/usr/bin/env bash
set -euo pipefail

cd /home/boyuanjie/projects/tasa-droid

PYTHON_BIN="${CARE_ML_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
OUT="artifacts/care-di25-hostego-future-v1"
M1_OUT="$OUT/m1_hostego_anomaly"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "BLOCKED_BY_CONTRACT: ML Python is not executable: $PYTHON_BIN" >&2
  exit 2
fi

echo "ML Python: $PYTHON_BIN"
"$PYTHON_BIN" -c 'import numpy, sklearn, torch, yaml; print("Dependencies:", numpy.__version__, sklearn.__version__, torch.__version__, "cuda=" + str(torch.cuda.is_available()))'
"$PYTHON_BIN" scripts/care-di25-hostego-future-v1/selftest_future_hostego.py
"$PYTHON_BIN" scripts/care-di25-hostego-future-v1/preflight_future.py

mkdir -p "$M1_OUT"

if "$PYTHON_BIN" -c 'import csv,json,pathlib; p=pathlib.Path("artifacts/care-di25-hostego-future-v1/m1_hostego_anomaly/future_m1_predictions_label_free.csv"); q=p.with_name("protocol.json"); rows=list(csv.DictReader(p.open(encoding="utf-8-sig"))) if p.is_file() else []; protocol=json.loads(q.read_text(encoding="utf-8")) if q.is_file() else {}; assert len(rows)==1989 and len({r["sha256"].strip().lower() for r in rows})==1989 and protocol.get("source_only_training") is True and protocol.get("future_labels_loaded_by_model") is False and protocol.get("parameter_count")==34613' 2>/dev/null; then
  echo "[resume] frozen label-free M1 future probabilities already complete; training skipped"
else
  "$PYTHON_BIN" scripts/care-di25-hostego-future-v1/run_m1_future.py \
    --device auto \
    --max-delta 2.25 \
    --seeds 900000,900001,900002
fi

"$PYTHON_BIN" scripts/care-di25-hostego-future-v1/summarize_future.py
cat "$OUT/results/REPORT.txt"

echo "DI25_HOSTEGO_FUTURE_DIAGNOSTIC_DONE"
