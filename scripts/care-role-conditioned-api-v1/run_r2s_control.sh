#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY=${PYTHON_BIN:-/home/boyuanjie/envs/tif/bin/python};S=scripts/care-role-conditioned-api-v1;R=artifacts/care-role-conditioned-api-v1;mkdir -p "$R/logs" "$R/arms/R2S_FIXED_SHUFFLED"
"$PY" "$S/selftest.py";"$PY" "$S/preflight.py"
if [[ ! -s "$R/cache/role_api_raw_source.npz" ]];then "$PY" "$S/build_cache.py";fi
if [[ ! -s "$R/arms/R2_FIXED_ROLE_API/source_predictions.csv" ]];then echo 'BLOCKED_BY_CONTRACT: completed R2 result is required';exit 1;fi
if [[ ! -s "$R/arms/R2S_FIXED_SHUFFLED/source_predictions.csv" ]];then CUDA_VISIBLE_DEVICES=0 "$PY" "$S/run_arm.py" --arm R2S_FIXED_SHUFFLED --device cuda --seeds 0,1,2 >"$R/logs/R2S_FIXED_SHUFFLED.gpu0.log" 2>&1;else echo '[resume] R2S_FIXED_SHUFFLED';fi
"$PY" "$S/summarize_r2s.py"
