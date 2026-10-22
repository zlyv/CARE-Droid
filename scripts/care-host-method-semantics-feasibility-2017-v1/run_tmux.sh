#!/usr/bin/env bash
# Run the complete read-only 2017 Host method-semantics feasibility audit in a detached tmux session.
# Inputs are frozen CARE assets and original APKs; outputs go only to the new v1 artifact directory.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
PROJECT=/home/boyuanjie/projects/tasa-droid
SCRIPT="$PROJECT/scripts/care-host-method-semantics-feasibility-2017-v1"
OUTPUT="$PROJECT/artifacts/care_host_method_semantics_feasibility_2017_v1"
PYTHON=/home/boyuanjie/envs/tif/bin/python
cd "$PROJECT"
mkdir -p "$OUTPUT"
date -Is > "$OUTPUT/tmux_started_at.txt"
{
  echo "HOST_METHOD_SEMANTICS_FULL_RUN_START $(date -Is)"
  "$PYTHON" "$SCRIPT/run.py" --full
  echo "HOST_METHOD_SEMANTICS_FULL_RUN_END $(date -Is)"
} 2>&1 | tee -a "$OUTPUT/experiment.log"

