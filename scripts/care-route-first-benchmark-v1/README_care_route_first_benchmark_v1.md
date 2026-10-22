# CARE RGCondAttn route-first benchmark v1

This benchmark trains one frozen RGCondAttn model on the established 2017
source cohort and compares two inference executions on the identical 1,989-APK
future cohort.  It alternates execution order across repeats and requires the
final probabilities to match within `1e-6`.

It measures cached model-side inference.  It does not claim raw-APK extraction
or end-to-end energy savings.

Run in tmux from the repository root:

```bash
cd /home/boyuanjie/projects/tasa-droid

OUT=artifacts/care-rgcondattn-route-first-benchmark-v1
mkdir -p "$OUT"

/home/boyuanjie/envs/tif/bin/python \
  scripts/benchmark_care_rgcondattn_route_first.py \
  --device cuda \
  --seed 0 \
  --repeats 5 \
  --output "$OUT" \
  2>&1 | tee "$OUT/run.log"
```

Return these files:

```bash
cat "$OUT/REPORT.txt"
cat "$OUT/inference_repeats.csv"
cat "$OUT/cost_summary.json"
```
