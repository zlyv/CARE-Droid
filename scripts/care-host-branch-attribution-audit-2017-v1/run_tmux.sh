#!/usr/bin/env bash
# Launch the frozen 2017 OOF Critical/Attended Host attribution audit.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=1
ROOT=/home/boyuanjie/projects/tasa-droid
HERE="$ROOT/scripts/care-host-branch-attribution-audit-2017-v1"
OUT="$ROOT/artifacts/care_host_branch_attribution_audit_2017_v2"
PY=/home/boyuanjie/envs/tif/bin/python
cd "$ROOT"
if [ -e "$OUT" ]; then echo "Refusing to overwrite $OUT" >&2; exit 2; fi
mkdir -p "$OUT/logs" "$OUT/config" "$OUT/integrity" "$OUT/features" "$OUT/interventions" "$OUT/groups" "$OUT/statistics" "$OUT/reports"
cat > "$OUT/logs/run_status.json" <<EOF
{"status":"RUNNING","stage":"launcher","started_at":"$(date --iso-8601=seconds)","pid":$$}
EOF
exec > >(tee -a "$OUT/logs/experiment.log") 2>&1
echo "BRANCH_AUDIT_RECOVERY_START $(date --iso-8601=seconds)"
if "$PY" "$HERE/finalize.py"; then
  echo "BRANCH_AUDIT_END $(date --iso-8601=seconds)"
else
  code=$?
  "$PY" - "$OUT/logs/run_status.json" "$code" <<'PY'
import json,sys,time
with open(sys.argv[1],'w',encoding='utf-8') as f:json.dump({'status':'FAILED','stage':'exception','exit_code':int(sys.argv[2]),'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')},f,indent=2)
PY
  exit "$code"
fi
