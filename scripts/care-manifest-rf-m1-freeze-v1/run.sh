#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd /home/boyuanjie/projects/tasa-droid
exec /home/boyuanjie/envs/tif/bin/python scripts/care-manifest-rf-m1-freeze-v1/freeze.py
