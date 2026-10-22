#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_EXE="$PYTHON_BIN"
elif [[ -x /home/boyuanjie/envs/tif/bin/python ]]; then
  PYTHON_EXE=/home/boyuanjie/envs/tif/bin/python
else
  PYTHON_EXE="$(command -v python3)"
fi

cat artifacts/care-source-recovery-stage2only-v1/smoke/REPORT.txt
"$PYTHON_EXE" scripts/care-source-recovery-stage2only-v1/smoke_approval.py approve
echo "Now rerun: bash scripts/care-source-recovery-stage2only-v1/launch_tmux.sh"
