#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
bash scripts/care-malben-host-v2/run_all_gates_on_server.sh
export CARE_GATES_PRECHECKED=1
bash scripts/care-malben-host-v2/launch_selective_tmux.sh
bash scripts/care-stage2only-malben-v1/launch_tmux.sh
echo "Both predeclared source-only experiments are running in separate tmux sessions."
