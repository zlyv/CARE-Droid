# CARE-Droid reliability-calibrated evidential router v2

This version calibrates conflict-aware evidential uncertainty toward the
continuous Stage-I decision risk

```text
r_i = abs(y_i - p_i^(1)).
```

The target is near zero for a confident correct prediction, near 0.5 for a
boundary prediction, and near one for a confident mistake.  It therefore
places boundary cases and high-confidence errors on one risk axis without
manually adding A25 and conflict scores.

## Loss and fixed settings

Four Manifest/Code/Sensitive/Graph evidence heads and conflict-aware fusion are
retained.  The primary objective is

```text
L = mean_v L_EDL(v)
    + lambda_c * Huber(U, r)
    + lambda_r * WeightedPairRank(U, r).
```

For each sampled pair with `r_i > r_j + 0.05`, the ranking term is

```text
(r_i-r_j) * softplus(-(U_i-U_j)).
```

Pair weights are normalized by their mean.  At most 8192 deterministic pairs
are used per fit.  Defaults are fixed before future inspection:

- `lambda_c=0.50`;
- `lambda_r=0.10`;
- Huber beta `0.10`;
- 300 epochs and a 25% route budget.

These are engineering defaults, not future-tuned optimal values.

## Strict nested evaluation

For an outer 2017 validation fold, the continuous risks used to train the
router are generated only inside the remaining four folds by inner OOF
Global24 models.  The outer fold's labels never participate in router fitting
or risk-target construction.  The package compares:

- `A25`;
- `EVI_ConflictMean25`: `lambda_c=lambda_r=0`;
- `RC_EVI_CalOnly25`: continuous Huber calibration only;
- `RC_EVI_CalRank25`: calibration plus ranking, the primary route;
- `RC_EVI_CalRank_Ignorance25`: removes the conflict part at routing time.

The 2018--2021 diagnostic trains on all 2017 rows using frozen 2017 OOF
Global24 probabilities.  Future labels are read only after predictions and
routes freeze.

## Bounded correction versus direct second-layer decision

The source audit applies each route to two independently defined second layers:

- `BoundedCorrection`: the established frozen bounded residual corrector;
- `DirectL2`: an independent RGCondAttn classifier trained with ordinary BCE,
  excluding Global24 features and its probability/logit.

They receive exactly the same routed APKs.  Thus the comparison changes the
second-layer decision mechanism, not the routing budget.  If Direct-L2 OOF
predictions are absent, the launcher first runs the existing strict source
ablation `care-rgcondattn-direct-l2-source-v1`.

Direct-L2 remains source-gated in this package.  It is not trained or selected
on the already-inspected future labels.  Only the bounded corrector is used in
the frozen future router diagnostic.

## Dependencies

The following previously supplied directories must be under `scripts/`:

```text
care-evidential-conflict-router-v1
care-multiview-conformal-router-source-v1
care-malcertain-router-future-v1
care-rgcondattn-direct-l2-source-v1
care-rgcondattn-dedup-bounded-source-v1
care-rgcondattn-dedup-bounded-future-v1
care-rolegraph-conditioned-v4-script
```

## Run

```bash
bash scripts/care-reliability-calibrated-evidential-router-v2/run_on_server.sh \
  2>&1 | tee artifacts/care-reliability-calibrated-evidential-router-v2.console.log
```

## Package results

```bash
zip -r care-reliability-calibrated-evidential-router-results-v2.zip \
  artifacts/care-reliability-calibrated-evidential-router-source-v2 \
  artifacts/care-reliability-calibrated-evidential-router-future-v2 \
  artifacts/care-rgcondattn-direct-l2-source-v1 \
  artifacts/care-reliability-calibrated-evidential-router-v2.console.log
```
