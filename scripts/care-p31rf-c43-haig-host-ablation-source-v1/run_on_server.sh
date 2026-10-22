#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python
S=scripts/care-p31rf-c43-haig-host-ablation-source-v1
R=artifacts/care-p31rf-c43-haig-host-ablation-source-v1
mkdir -p "$R/arms" "$R/logs" "$R/audits"
"$PY" "$S/selftest.py"
"$PY" "$S/preflight.py"
"$PY" "$S/audit_alignment.py"
"$PY" "$S/build_caches.py"
run(){ local arm="$1" gpu="$2"; if [[ -s "$R/arms/$arm/source_predictions.csv" ]];then echo "[resume] $arm";return;fi; echo "[run] $arm GPU$gpu";CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$S/run_arm.py" --arm "$arm" --device cuda --seeds 0,1,2 >"$R/logs/$arm.gpu$gpu.log" 2>&1;echo "[done] $arm GPU$gpu";}
run A0_ROLE_ONLY 0 & p0=$!;run A2_H3_NO_ROLE 1 & p1=$!;status=0;wait $p0||status=1;wait $p1||status=1;[[ $status == 0 ]]||{ echo BLOCKED_BY_CONTRACT: phase1 failed;exit 1;}
run A1_API_NO_ROLE 0 & p0=$!;run A3_ROLE_PLUS_H3 1 & p1=$!;status=0;wait $p0||status=1;wait $p1||status=1;[[ $status == 0 ]]||{ echo BLOCKED_BY_CONTRACT: phase2 failed;exit 1;}
run A4_H3_SHUFFLE_NO_ROLE 0
"$PY" "$S/summarize.py"
echo P31RF_C43_HAIG_HOST_ABLATION_FULL_RUN_DONE

