#!/usr/bin/env bash
# Runs only this isolated committee experiment; no APK or historical writes.
set -euo pipefail
: "${MFD_OUT:?Set a fresh isolated MFD_OUT run directory}"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$MFD_OUT/tmp" XDG_CACHE_HOME="$MFD_OUT/cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd /home/boyuanjie/projects/tasa-droid
/home/boyuanjie/envs/tif/bin/python -B scripts/care-manifest-four-member-disagreement-v1/run.py >> "$MFD_OUT/run.log" 2>&1
