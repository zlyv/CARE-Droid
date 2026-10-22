#!/usr/bin/env bash
# Create a fresh isolated output and run the conditional Host-path pilot in tmux.
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
test ! -e outputs/care_host_region_attribution_v4
mkdir outputs/care_host_region_attribution_v4
mkdir -m 700 outputs/care_host_region_attribution_v4/tmp
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/home/boyuanjie/projects/tasa-droid/outputs/care_host_region_attribution_v4/tmp
export XDG_CACHE_HOME=/home/boyuanjie/projects/tasa-droid/outputs/care_host_region_attribution_v4/cache
CUDA_VISIBLE_DEVICES=1 tmux new-session -d -s care-host-region-attribution-v4 "/home/boyuanjie/envs/tif/bin/python scripts/care-host-region-attribution-v4/run.py --pilot --device cuda > outputs/care_host_region_attribution_v4/run.log 2>&1"
echo 'Started: care-host-region-attribution-v4'
echo 'Monitor: tail -F /home/boyuanjie/projects/tasa-droid/outputs/care_host_region_attribution_v4/run.log'
