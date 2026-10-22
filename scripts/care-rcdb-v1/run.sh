#!/usr/bin/env bash
# RCDB-V1 autonomous runner; writes only RCDB_OUTPUT; never reads or runs APKs.
set -Eeuo pipefail
ROOT=/home/boyuanjie/projects/tasa-droid
CODE="$ROOT/scripts/care-rcdb-v1"
: "${RCDB_OUTPUT:?RCDB_OUTPUT must be set by launch.sh}"
: "${RCDB_DEVICE:=cuda}"
export PYTHONDONTWRITEBYTECODE=1
export RCDB_OUTPUT RCDB_DEVICE
mkdir -p "$RCDB_OUTPUT"
exec > >(tee -a "$RCDB_OUTPUT/run.log") 2>&1
phase=bootstrap
started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
finish() {
  rc=$?
  trap - EXIT
  printf '%s\n' "$rc" > "$RCDB_OUTPUT/exit_code.txt"
  if [ "$rc" -eq 0 ]; then
    printf '{"status":"COMPLETE","phase":"%s","started_utc":"%s"}\n' "$phase" "$started" > "$RCDB_OUTPUT/run_state.json"
    touch "$RCDB_OUTPUT/completed.flag"
  else
    printf '{"status":"FAILED","phase":"%s","exit_code":%s,"started_utc":"%s"}\n' "$phase" "$rc" "$started" > "$RCDB_OUTPUT/run_state.json"
  fi
}
trap finish EXIT
step() {
  phase="$1"
  printf '{"status":"RUNNING","phase":"%s","started_utc":"%s"}\n' "$phase" "$started" > "$RCDB_OUTPUT/run_state.json"
  printf '{"phase":"%s","status":"RUNNING"}\n' "$phase" > "$RCDB_OUTPUT/phase_status.json"
  echo "RCDB_PHASE_START $phase $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  shift
  "$@"
  printf '{"phase":"%s","status":"DONE"}\n' "$phase" > "$RCDB_OUTPUT/phase_status.json"
  echo "RCDB_PHASE_DONE $phase $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
cd "$ROOT"
PY=/home/boyuanjie/envs/tif/bin/python
step preflight "$PY" "$CODE/main.py" --audit-only --device "$RCDB_DEVICE" ${RCDB_RESUME:+--resume}
step train_oof "$PY" "$CODE/main.py" --train-oof --device "$RCDB_DEVICE" --resume
step source_eval "$PY" "$CODE/main.py" --source-eval --device "$RCDB_DEVICE" --resume
step train_final2017 "$PY" "$CODE/main.py" --train-final2017 --device "$RCDB_DEVICE" --resume
step evaluate_future "$PY" "$CODE/main.py" --evaluate-future --device "$RCDB_DEVICE" --resume
step collect "$PY" "$CODE/main.py" --collect --device "$RCDB_DEVICE" --resume
phase=complete
