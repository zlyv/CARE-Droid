#!/usr/bin/env bash
# HDBC-V1 unattended pipeline; writes only HDBC_OUTPUT and never removes files.
set -euo pipefail
ROOT=/home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python
export PYTHONDONTWRITEBYTECODE=1
export HDBC_OUTPUT="${HDBC_OUTPUT:-$ROOT/outputs/care_hdbc_v1}"
DEVICE="${HDBC_DEVICE:-cuda}"
mkdir -p "$HDBC_OUTPUT"
exec > >(tee -a "$HDBC_OUTPUT/run.log") 2>&1
phase=initial
on_exit() {
  rc=$?
  printf '%s\n' "$rc" > "$HDBC_OUTPUT/exit_code.txt"
  if ((rc==0)); then
    printf '{"status":"COMPLETE","phase":"%s","exit_code":0}\n' "$phase" > "$HDBC_OUTPUT/run_state.json"
    printf 'COMPLETE\n' > "$HDBC_OUTPUT/completed.flag"
  else
    printf '{"status":"FAILED","phase":"%s","exit_code":%s}\n' "$phase" "$rc" > "$HDBC_OUTPUT/run_state.json"
    echo "HDBC_PIPELINE_FAILED phase=$phase rc=$rc"
  fi
}
trap on_exit EXIT
resume=()
if [[ "${HDBC_RESUME:-0}" == 1 ]]; then resume=(--resume); fi
for phase in audit-only extract-features train-source evaluate-source train-final2017 evaluate-future collect; do
  marker="$HDBC_OUTPUT/phases/$phase.done"
  if [[ -e "$marker" ]]; then
    echo "HDBC_PHASE_RESUME name=$phase"
    resume=(--resume)
    continue
  fi
  printf '{"phase":"%s","status":"RUNNING","started_utc":"%s"}\n' "$phase" "$(date -u +%FT%TZ)" > "$HDBC_OUTPUT/phase_status.json"
  echo "HDBC_PHASE_START name=$phase epoch=$(date +%s)"
  "$PY" "$ROOT/scripts/care-hdbc-v1/main.py" "--$phase" --device "$DEVICE" --workers 1 "${resume[@]}"
  mkdir -p "$HDBC_OUTPUT/phases"
  printf 'DONE\n' > "$marker"
  printf '{"phase":"%s","status":"DONE","completed_utc":"%s"}\n' "$phase" "$(date -u +%FT%TZ)" > "$HDBC_OUTPUT/phase_status.json"
  echo "HDBC_PHASE_DONE name=$phase epoch=$(date +%s)"
  resume=(--resume)
done
echo "HDBC_PIPELINE_COMPLETE output=$HDBC_OUTPUT"
