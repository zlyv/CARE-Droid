#!/usr/bin/env bash
# Create a fresh isolated output and run the conditional Host-path pilot in tmux.
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
test ! -e outputs/care_host_region_attribution_v6
mkdir outputs/care_host_region_attribution_v6
mkdir -m 700 outputs/care_host_region_attribution_v6/tmp
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/home/boyuanjie/projects/tasa-droid/outputs/care_host_region_attribution_v6/tmp
export XDG_CACHE_HOME=/home/boyuanjie/projects/tasa-droid/outputs/care_host_region_attribution_v6/cache
tmux new-session -d -s care-host-region-attribution-v6 "env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/home/boyuanjie/projects/tasa-droid/outputs/care_host_region_attribution_v6/tmp XDG_CACHE_HOME=/home/boyuanjie/projects/tasa-droid/outputs/care_host_region_attribution_v6/cache OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/boyuanjie/envs/tif/bin/python scripts/care-host-region-attribution-v6/run.py --pilot --device cpu > outputs/care_host_region_attribution_v6/run.log 2>&1"
echo 'Started: care-host-region-attribution-v6'
echo 'Monitor: tail -F /home/boyuanjie/projects/tasa-droid/outputs/care_host_region_attribution_v6/run.log'
