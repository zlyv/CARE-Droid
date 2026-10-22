#!/usr/bin/env bash
# Launch the read-only 2017 OOF Stage-II error/complementarity audit.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=1
ROOT=/home/boyuanjie/projects/tasa-droid
HERE="$ROOT/scripts/care-stage2-error-complementarity-audit-2017-v1"
OUT="$ROOT/artifacts/care_stage2_error_complementarity_audit_2017_v3"
PY=/home/boyuanjie/envs/tif/bin/python
cd "$ROOT"
if [ -e "$OUT" ]; then
  echo "Refusing to overwrite existing output: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT/logs"
cat > "$OUT/logs/run_status.json" <<EOF
{"status":"RUNNING","stage":"launcher","started_at":"$(date --iso-8601=seconds)","pid":$$}
EOF
exec > >(tee -a "$OUT/logs/experiment.log") 2>&1
echo "AUDIT_START $(date --iso-8601=seconds)"
if "$PY" "$HERE/audit.py"; then
  echo "AUDIT_END $(date --iso-8601=seconds)"
else
  code=$?
  "$PY" - "$OUT/logs/run_status.json" "$code" <<'PY'
import json,sys,time
p=sys.argv[1]; code=int(sys.argv[2])
with open(p,'w',encoding='utf-8') as f: json.dump({'status':'FAILED','stage':'exception','exit_code':code,'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')},f,ensure_ascii=False,indent=2)
PY
  exit "$code"
fi
