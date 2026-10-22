#!/usr/bin/env bash
# Safe first phase of expanded-2017 CARE-Droid study; no model fit or APK execution.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/home/boyuanjie/projects/tasa-droid/scripts/care-expanded2017-crossyear-v1/tmp
export XDG_CACHE_HOME="$TMPDIR/cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd /home/boyuanjie/projects/tasa-droid
/home/boyuanjie/envs/tif/bin/python scripts/care-expanded2017-crossyear-v1/audit_datasets.py --audit-only
