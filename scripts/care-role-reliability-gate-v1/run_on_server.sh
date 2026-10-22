#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python
S=scripts/care-role-reliability-gate-v1
A=artifacts/care-role-reliability-gate-v1
mkdir -p "$A/logs" "$A/nested"
export PYTHONWARNINGS=ignore
export LOGURU_LEVEL=ERROR
"$PY" "$S/selftest.py"
"$PY" "$S/audit_existing.py"
if "$PY" - <<'PY'
import json
d=json.load(open('artifacts/care-role-reliability-gate-v1/existing_experiment_audit.json'))
raise SystemExit(0 if d['status']=='FOUND' and d['equivalent'] else 1)
PY
then
  echo ROLE_RELIABILITY_REUSE_EXISTING_RESULT
  exit 0
fi
"$PY" "$S/write_protocol.py"
"$PY" "$S/preflight.py"
"$PY" "$S/build_features.py"
"$PY" "$S/feature_audit.py"
if "$PY" - <<'PY'
import json
d=json.load(open('artifacts/care-role-reliability-gate-v1/diagnostic_decision.json'))
raise SystemExit(0 if d['advance_to_strict_nested_gate'] else 1)
PY
then
  run_pair() {
    local f0="$1" f1="$2"
    echo "[run] strict nested outer $f0 on GPU0"
    CUDA_VISIBLE_DEVICES=0 "$PY" "$S/nested_replay.py" --outer-fold "$f0" --device cuda >"$A/logs/nested_outer_${f0}.gpu0.log" 2>&1 & local p0=$!
    if [[ "$f1" != "0" ]]; then echo "[run] strict nested outer $f1 on GPU1"; CUDA_VISIBLE_DEVICES=1 "$PY" "$S/nested_replay.py" --outer-fold "$f1" --device cuda >"$A/logs/nested_outer_${f1}.gpu1.log" 2>&1 & local p1=$!; wait "$p0"; wait "$p1"; echo "[done] strict nested outer $f0,$f1"; else wait "$p0"; echo "[done] strict nested outer $f0"; fi
  }
  run_pair 1 2
  run_pair 3 4
  run_pair 5 0
  "$PY" "$S/run_gate.py"
else
  echo ROLE_RELIABILITY_DIAGNOSTIC_STOP_NO_NESTED_TRAINING
fi
"$PY" "$S/summarize.py"
echo CARE_ROLE_RELIABILITY_SOURCE_PIPELINE_DONE
echo artifacts=$A
echo final_summary=$A/final_summary.json

