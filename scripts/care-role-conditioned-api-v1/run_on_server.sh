#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY=${PYTHON_BIN:-/home/boyuanjie/envs/tif/bin/python}
S=scripts/care-role-conditioned-api-v1
R=artifacts/care-role-conditioned-api-v1
mkdir -p "$R/arms" "$R/logs" "$R/audits"
export PYTHONHASHSEED=0
"$PY" "$S/selftest.py"
"$PY" "$S/preflight.py"
"$PY" "$S/feasibility.py"
"$PY" "$S/a0_a3_overlap.py"
if [[ ! -s "$R/cache/role_api_raw_source.npz" || ! -s "$R/cache/role_api_registry.npz" ]];then "$PY" "$S/build_cache.py";else echo '[resume] role API cache';fi
run(){ local arm="$1" gpu="$2";if [[ -s "$R/arms/$arm/source_predictions.csv" ]];then echo "[resume] $arm";return;fi;echo "[run] $arm GPU$gpu";CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$S/run_arm.py" --arm "$arm" --device cuda --seeds 0,1,2 >"$R/logs/$arm.gpu$gpu.log" 2>&1;echo "[done] $arm GPU$gpu";}
run R1_UNCONDITIONAL_API 0 & p0=$!;run R2_FIXED_ROLE_API 1 & p1=$!;status=0;wait $p0||status=1;wait $p1||status=1;[[ $status == 0 ]]||{ echo 'BLOCKED_BY_CONTRACT: R1/R2 failed';exit 1;}
run R3_LEARNED_ROLE_API 0 & p0=$!;run R4_SHUFFLED_WITHIN_ROLE 1 & p1=$!;status=0;wait $p0||status=1;wait $p1||status=1;[[ $status == 0 ]]||{ echo 'BLOCKED_BY_CONTRACT: R3/R4 failed';exit 1;}
echo R0_DONE;echo R1_DONE;echo R2_DONE;echo R3_DONE;echo R4_DONE
"$PY" "$S/run_count_control.py"
"$PY" "$S/summarize.py"
