#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=1
cd /home/boyuanjie/projects/tasa-droid/scripts/care-cfg-evidence-audit-2017-v3
PY=/home/boyuanjie/envs/tif/bin/python
$PY integrity.py pre
$PY static_audit.py
$PY model_audit.py
$PY finalize.py
$PY integrity.py post
echo CARE_CFG_EVIDENCE_AUDIT_COMPLETE
