#!/usr/bin/env bash
set -euo pipefail

cd /home/boyuanjie/projects/tasa-droid

PYTHON_BIN="${CARE_ML_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
OUT="artifacts/care-di25-hostego-within-year-v1"
M1_OUT="$OUT/m1_hostego_anomaly_bounded"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "BLOCKED_BY_CONTRACT: ML Python is not executable: $PYTHON_BIN" >&2
  exit 2
fi

echo "ML Python: $PYTHON_BIN"
"$PYTHON_BIN" -c 'import numpy, sklearn, torch, yaml; print("Dependencies:", numpy.__version__, sklearn.__version__, torch.__version__, "cuda=" + str(torch.cuda.is_available()))'
"$PYTHON_BIN" scripts/care-di25-hostego-within-year-v1/selftest_within_year_hostego.py
"$PYTHON_BIN" scripts/care-di25-hostego-within-year-v1/preflight_within_year.py

mkdir -p "$M1_OUT"
if "$PYTHON_BIN" -c 'import csv,json,pathlib; p=pathlib.Path("artifacts/care-di25-hostego-within-year-v1/m1_hostego_anomaly_bounded/within_year_m1_predictions.csv"); q=p.with_name("protocol.json"); rows=list(csv.DictReader(p.open(encoding="utf-8-sig"))) if p.is_file() else []; d=json.loads(q.read_text(encoding="utf-8")) if q.is_file() else {}; assert len(rows)==1989 and len({r["sha256"].strip().lower() for r in rows})==1989 and d.get("protocol")=="independent per-year five-fold OOF" and d.get("parameter_count")==34613 and d.get("version")=="hostego-anomaly-bounded-sameyear-v1"' 2>/dev/null; then
  echo "[resume] complete frozen M1 same-year OOF predictions already exist; training skipped"
else
  "$PYTHON_BIN" scripts/care-di25-hostego-within-year-v1/run_m1_within_year.py \
    --device auto \
    --max-delta 2.25 \
    --seeds 900000,900001,900002
fi

"$PYTHON_BIN" scripts/care-di25-hostego-within-year-v1/summarize_within_year.py
cat "$OUT/results/REPORT.txt"
echo "DI25_HOSTEGO_WITHIN_YEAR_DIAGNOSTIC_DONE"
