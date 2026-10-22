#!/usr/bin/env bash
# Launch the read-only audit in an independent tmux session; writes new output only.
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
test ! -e outputs/care_host_region_attribution_v1
test ! -e outputs/care_host_region_attribution_v1.console.log
test ! -e outputs/care_host_region_attribution_v1.log
export PYTHONDONTWRITEBYTECODE=1
tmux new-session -d -s care-host-region-attribution-v1 "/home/boyuanjie/envs/tif/bin/python scripts/care-host-region-attribution-v1/audit.py --audit-only > outputs/care_host_region_attribution_v1.log 2>&1"
echo 'Started: care-host-region-attribution-v1'
echo 'Monitor: tail -F /home/boyuanjie/projects/tasa-droid/outputs/care_host_region_attribution_v1.log'
