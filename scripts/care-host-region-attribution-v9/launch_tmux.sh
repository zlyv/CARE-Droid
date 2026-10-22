#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
test ! -e outputs/care_host_region_attribution_v9
mkdir outputs/care_host_region_attribution_v9
mkdir -m 700 outputs/care_host_region_attribution_v9/tmp
tmux new-session -d -s care-host-region-attribution-v9 "env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/home/boyuanjie/projects/tasa-droid/outputs/care_host_region_attribution_v9/tmp XDG_CACHE_HOME=/home/boyuanjie/projects/tasa-droid/outputs/care_host_region_attribution_v9/cache /home/boyuanjie/envs/tif/bin/python scripts/care-host-region-attribution-v9/map_flip_exceptions.py > outputs/care_host_region_attribution_v9/run.log 2>&1"
echo 'Started: care-host-region-attribution-v9'
echo 'Monitor: tail -F /home/boyuanjie/projects/tasa-droid/outputs/care_host_region_attribution_v9/run.log'
