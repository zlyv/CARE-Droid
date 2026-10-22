#!/usr/bin/env bash
# Independent, resumable Host-Role contrastive development experiment.
# All new files are written under the new output directory. No APK is read.
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
export PYTHONDONTWRITEBYTECODE=1
OUT=outputs/care_benign_host_role_contrastive_v1
CODE=scripts/care-benign-host-role-contrastive-v1
PY=/home/boyuanjie/envs/tif/bin/python
mkdir -p "$OUT/logs"
assets=(
  outputs/care_manifest_four_member_disagreement_v1/run_20261019_pilot01/per_apk_predictions.csv
  scripts/care-existing-three-background-contrastive-v2/contrastive.py
  scripts/care-existing-three-background-contrastive-v2/common.py
  scripts/care-multiyear-development-v1/train_stage2.py
)
for tag in C0R_2017 T2_C1 T3_C1 T4_C1; do
  for seed in 900000 900001 900002; do
    assets+=("outputs/care_multiyear_development_v1/models/stage2/$tag/$seed.pkl")
  done
  assets+=("outputs/care_multiyear_development_v1/predictions/$tag/expert.npz")
done
if [[ ! -e "$OUT/asset_sha256_before.txt" ]]; then
  sha256sum "${assets[@]}" > "$OUT/asset_sha256_before.txt"
else
  sha256sum -c "$OUT/asset_sha256_before.txt" > "$OUT/logs/asset_sha256_resume_check.log"
fi
echo "BHRC_PIPELINE_START output=$OUT gpu=${CUDA_VISIBLE_DEVICES:-unspecified}"
for tag in C0R_2017 T2_C1 T3_C1 T4_C1; do
  for seed in 900000 900001 900002; do
    for arm in H1_TRUE_ROLE H2_ROLE_SHUFFLE; do
      base="$OUT/full/$tag/$seed/$arm"
      if [[ -f "${base}_predictions.npz" && -f "${base}_audit.json" && -f "${base}_checkpoint.pt" ]]; then
        echo "BHRC_ARM_RESUME $tag $seed $arm"
        continue
      fi
      if [[ -e "${base}_predictions.npz" || -e "${base}_audit.json" || -e "${base}_checkpoint.pt" ]]; then
        echo "BHRC_PARTIAL_ARM_REQUIRES_MANUAL_REVIEW $tag $seed $arm" >&2
        exit 1
      fi
      echo "BHRC_ARM_START $tag $seed $arm"
      "$PY" "$CODE/run.py" --tag "$tag" --seed "$seed" --arm "$arm" --device cuda \
        > "$OUT/logs/${tag}_${seed}_${arm}.log" 2>&1
      tail -n 2 "$OUT/logs/${tag}_${seed}_${arm}.log"
    done
  done
done
if [[ ! -e "$OUT/final_decision.json" ]]; then
  "$PY" "$CODE/summarize.py" > "$OUT/logs/summarize.log" 2>&1
fi
sha256sum -c "$OUT/asset_sha256_before.txt" > "$OUT/logs/asset_sha256_after_check.log"
echo "BHRC_INTEGRITY_PASS"
echo "BHRC_PIPELINE_DONE output=$OUT"
