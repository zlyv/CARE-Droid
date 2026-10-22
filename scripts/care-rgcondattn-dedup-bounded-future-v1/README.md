# Deduplicated bounded RGCondAttn future diagnostic v1

This runs the source-frozen deduplicated bounded RGCondAttn on 2018--2021.
These years have already been inspected in the project, so the results are
diagnostic-only.  Future labels are materialized only after all probabilities
and the untuned equal-weight three-seed ensemble are fixed.

Dependencies already expected under `scripts/`:

- CARE v3/v4 source scripts;
- CARE v4 frozen future script;
- `care-rgcondattn-dedup-bounded-source-v1`.

The script also loads the established original RGCondAttn future predictions
when available and emits a paired comparison.  Override its path with the
`ORIGINAL_FUTURE_RESULT` environment variable if needed.

Run:

```bash
bash scripts/care-rgcondattn-dedup-bounded-future-v1/run_on_server.sh \
  2>&1 | tee artifacts/care-rgcondattn-dedup-bounded-future-v1.console.log
```

If training and prediction already finished but the summary step failed, rerun
only the idempotent post-processing stage (no model is retrained):

```bash
/home/boyuanjie/envs/tif/bin/python \
  scripts/care-rgcondattn-dedup-bounded-future-v1/run_care_rgcondattn_dedup_bounded_future_v1.py \
  --output artifacts/care-rgcondattn-dedup-bounded-future-diagnostic-v1 \
  --postprocess-only
```
