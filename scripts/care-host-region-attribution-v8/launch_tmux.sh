#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
test ! -e outputs/care_host_region_attribution_v8
mkdir outputs/care_host_region_attribution_v8
mkdir -m 700 outputs/care_host_region_attribution_v8/tmp
tmux new-session -d -s care-host-region-attribution-v8 "env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/home/boyuanjie/projects/tasa-droid/outputs/care_host_region_attribution_v8/tmp XDG_CACHE_HOME=/home/boyuanjie/projects/tasa-droid/outputs/care_host_region_attribution_v8/cache OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/boyuanjie/envs/tif/bin/python scripts/care-host-region-attribution-v8/map_and_report.py > outputs/care_host_region_attribution_v8/run.log 2>&1"
echo 'Started: care-host-region-attribution-v8'
echo 'Monitor: tail -F /home/boyuanjie/projects/tasa-droid/outputs/care_host_region_attribution_v8/run.log'
