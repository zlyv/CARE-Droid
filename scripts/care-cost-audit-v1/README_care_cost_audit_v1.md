# CARE-Droid cost audit v1

This package performs the first, training-free cost audit from the frozen CARE
v4 result files.  It answers two separate questions:

1. How much second-stage work could A25 avoid if routing happens first?
2. Did the current evaluation implementation actually avoid that work?

The script deliberately does **not** turn workload counts into invented runtime
numbers.  Wall-clock latency, RAM/VRAM, power, and raw-APK extraction must be
measured in the next benchmark after route-first execution is implemented.

Run from the repository root:

```bash
cd /home/boyuanjie/projects/tasa-droid

OUT=artifacts/care-cost-audit-v1

/home/boyuanjie/envs/tif/bin/python \
  scripts/audit_care_selective_cost.py \
  --future-result \
  artifacts/care-rolegraph-conditioned-v4-future-diagnostic-v1 \
  --source-result \
  artifacts/care-rolegraph-conditioned-dualstream-v4-full-seed0 \
  --artifact artifacts/p1-features/features.csv \
  --artifact artifacts/p25-endpoint-cache-full-cap256/endpoint_centered_tensor_cache.jsonl.gz \
  --artifact artifacts/clipped-intra-inter-structure-ablation-2017-v1-full/multiscale_features.npz \
  --artifact artifacts/s7-future-interz-cache-v1/future_interz.npz \
  --artifact artifacts/care-local-region-cache-v1/regions_2017_2021.npz \
  --output "$OUT" \
  2>&1 | tee "$OUT.console.log"
```

Expected completion marker:

```text
Done: artifacts/care-cost-audit-v1
```

Return `REPORT.txt`, `cost_by_year.csv`, `cost_summary.json`, and
`artifact_sizes.csv` for the next step.
