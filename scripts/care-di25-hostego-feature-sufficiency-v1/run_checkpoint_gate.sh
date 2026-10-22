#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
mkdir -p artifacts/care-di25-hostego-dmd-bounded-v1
/home/boyuanjie/envs/tif/bin/python \
  scripts/care-di25-hostego-feature-sufficiency-v1/audit_frozen_m1_checkpoints.py \
  2>&1 | tee artifacts/care-di25-hostego-dmd-bounded-v1/feature_sufficiency_checkpoint_gate.log
