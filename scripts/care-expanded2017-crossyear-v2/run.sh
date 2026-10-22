#!/usr/bin/env bash
# v2 phase-0 interpretation job; reads v1 evidence only.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd /home/boyuanjie/projects/tasa-droid
/home/boyuanjie/envs/tif/bin/python scripts/care-expanded2017-crossyear-v2/resolve_phase0.py
