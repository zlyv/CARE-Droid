#!/usr/bin/env bash
# Run new read-only graph reconstruction probe; writes only fresh v3 output.
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
test ! -e outputs/care_host_region_attribution_v3
test ! -e outputs/care_host_region_attribution_v3.log
export PYTHONDONTWRITEBYTECODE=1
tmux new-session -d -s care-host-region-attribution-v3 "/home/boyuanjie/envs/tif/bin/python scripts/care-host-region-attribution-v3/probe.py --probe-only > outputs/care_host_region_attribution_v3.log 2>&1"
echo 'Started: care-host-region-attribution-v3'
echo 'Monitor: tail -F /home/boyuanjie/projects/tasa-droid/outputs/care_host_region_attribution_v3.log'
