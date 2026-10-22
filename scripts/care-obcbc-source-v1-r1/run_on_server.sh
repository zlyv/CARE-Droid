#!/usr/bin/env bash
set -euo pipefail

PROJECT="${OBCBC_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OBCBC_OUTPUT:-$PROJECT/outputs/care-obcbc-source-v1}"

if [[ -x /home/boyuanjie/envs/tif/bin/python ]]; then
  PY=/home/boyuanjie/envs/tif/bin/python
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  echo "BLOCKED_BY_CONTRACT: no Python interpreter" >&2
  exit 1
fi

mkdir -p "$OUT/logs" "$OUT/checkpoints"
echo "OBCBC_PYTHON=$PY"
echo "CARE_OBCBC_STAGE_B_PIPELINE_START project=$PROJECT output=$OUT"

"$PY" -m py_compile "$SCRIPT_DIR"/*.py
"$PY" "$SCRIPT_DIR/selftest.py"
OBCBC_PROJECT="$PROJECT" OBCBC_OUTPUT="$OUT" PYTHONUNBUFFERED=1 \
  "$PY" "$SCRIPT_DIR/run_audit.py" 2>&1 | tee -a "$OUT/logs/run.log"

echo "CARE_OBCBC_STAGE_B_PIPELINE_DONE"
