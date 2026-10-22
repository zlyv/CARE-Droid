#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
test ! -e outputs/care_host_region_attribution_v11
mkdir outputs/care_host_region_attribution_v11
mkdir -m 700 outputs/care_host_region_attribution_v11/tmp
tmux new-session -d -s care-host-region-attribution-v11 "env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/home/boyuanjie/projects/tasa-droid/outputs/care_host_region_attribution_v11/tmp XDG_CACHE_HOME=/home/boyuanjie/projects/tasa-droid/outputs/care_host_region_attribution_v11/cache /home/boyuanjie/envs/tif/bin/python scripts/care-host-region-attribution-v11/map_decision_changers.py > outputs/care_host_region_attribution_v11/run.log 2>&1"
echo 'Started: care-host-region-attribution-v11'
echo 'Monitor: tail -F /home/boyuanjie/projects/tasa-droid/outputs/care_host_region_attribution_v11/run.log'
