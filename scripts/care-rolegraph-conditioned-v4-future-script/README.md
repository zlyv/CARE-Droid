# CARE-Droid v4 frozen future diagnostic

Copy all three files into the repository `scripts/` directory.  The existing
CARE v4 source script and all of its CARE/S7 dependencies must already be in
that same directory.

This run is frozen from the 2017 source decision:

- variants: `RGConcat,RGCondAttn`;
- seeds: `0,1,2`, exported separately and as an equal ensemble;
- years: `2018,2019,2020,2021`;
- no future label is used for training, epoch selection, routing, weighting,
  Inter-Z construction, or ensemble formation.

Run the structural self-test:

```bash
/home/boyuanjie/envs/tif/bin/python \
  scripts/selftest_care_rolegraph_future.py
```

Run the diagnostic in tmux:

```bash
cd /home/boyuanjie/projects/tasa-droid

OUT=artifacts/care-rolegraph-conditioned-v4-future-diagnostic-v1
mkdir -p "$OUT"

/home/boyuanjie/envs/tif/bin/python \
  scripts/run_care_rolegraph_conditioned_v4_future.py \
  --device cuda \
  --output "$OUT" \
  2>&1 | tee "$OUT/run.log"
```

Completion marker:

```text
Done: artifacts/care-rolegraph-conditioned-v4-future-diagnostic-v1
```

The output includes individual seed predictions and metrics, the equal
three-seed ensemble, comparisons with frozen Joint-S7 and CARE v3, and audit
files for the label-free future Inter-Z and local-region cache.
