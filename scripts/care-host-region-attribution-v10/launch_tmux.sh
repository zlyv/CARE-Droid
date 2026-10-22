#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
test ! -e outputs/care_host_region_attribution_v10
mkdir outputs/care_host_region_attribution_v10
mkdir -m 700 outputs/care_host_region_attribution_v10/tmp
tmux new-session -d -s care-host-region-attribution-v10 "env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/home/boyuanjie/projects/tasa-droid/outputs/care_host_region_attribution_v10/tmp /home/boyuanjie/envs/tif/bin/python scripts/care-host-region-attribution-v10/analyze_ensemble.py > outputs/care_host_region_attribution_v10/run.log 2>&1"
echo 'Started: care-host-region-attribution-v10'
echo 'Monitor: tail -F /home/boyuanjie/projects/tasa-droid/outputs/care_host_region_attribution_v10/run.log'
