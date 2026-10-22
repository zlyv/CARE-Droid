#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
SCRIPT=/home/boyuanjie/projects/tasa-droid/scripts/care-2020-fn-tn-dex-review-v4
OUT=/home/boyuanjie/projects/tasa-droid/outputs/care-2020-fn-tn-dex-review-v4
/usr/bin/python3 "$SCRIPT/preflight.py"
bwrap --unshare-net --ro-bind /usr /usr --ro-bind /etc /etc \
  --symlink usr/lib /lib --symlink usr/lib64 /lib64 \
  --dev-bind /dev /dev --proc /proc --tmpfs /tmp \
  --bind "$OUT/blind_workspace" /review --chdir /review \
  -- /usr/bin/python3 /review/blind_audit.py
/usr/bin/python3 "$SCRIPT/finish_preflight.py"
