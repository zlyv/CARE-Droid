# CARE-Droid error-aware evidential router v1

This experiment changes the meaning learned by evidential uncertainty.  The
same four evidence heads and conflict-aware fusion are retained, but their
unified uncertainty is additionally calibrated toward frozen Stage-I failure
risk.  Global24, the local CARE expert, the final threshold, and the 25% budget
remain frozen.

## Objective（训练目标）

For each APK, the four semantic views output nonnegative benign/malware
evidence.  Their ordinary evidential classification loss is retained.  Let
`z=1` when an out-of-fold Global24 prediction is wrong and `z=0` otherwise.
The jointly trained heads minimize:

```text
L = mean_v L_EDL(v) + lambda_r * BalancedBCE(U, z)
```

`U` is the same uncertainty produced by reliability discount and pairwise
conflict-aware evidence fusion.  There is no separate SVM or error-prediction
head.  Balanced BCE prevents the minority Stage-I error rows from being
overwhelmed by correct rows.

The default `lambda_r=0.25` is fixed before the future diagnostic.  The package
also trains a same-initialization `lambda_r=0` model, so the effect of changing
the training objective can be separated from architecture and randomness.

## Leakage control（严格嵌套）

In every 2017 outer fold, error labels for the outer-training rows are produced
by an inner OOF Global24 procedure.  A row is never used to train the model that
creates its own Stage-I error target.  The outer validation fold is not used in
head training, error-target construction, or route selection.

For the final 2018--2021 diagnostic, all four heads train on 2017 only.  The
Stage-I error targets use frozen 2017 OOF Global24 predictions.  Source
thresholds use strict 2017 outer-fold OOF router scores.  Future labels are
accessed only after scores, routes, and final probabilities are frozen.

## Compared routes

- `A25`: frozen Global24 boundary ambiguity;
- `EVI_ConflictMean25`: original joint EVI with `lambda_r=0`;
- `EA_EVI25`: primary error-aware unified evidential uncertainty;
- `EA_EVI_Ignorance25`: error-aware evidence insufficiency without conflict.

Every `*25` / `*YearTop25` method uses the same 25% Stage-II budget.  The future
audit also transports the 2017 Q75 score thresholds to reveal score-scale drift.

## Scientific boundary

Stage-I error is a practical proxy for "the first layer is unreliable."  It is
not identical to "the frozen Stage-II expert will correct this APK."  Therefore
this version may be called error-aware or reliability-aware evidential routing,
but should not be described as benefit-optimal routing.  A true benefit target
would require strictly nested Stage-II counterfactual predictions.

## Dependencies

Place these already supplied directories under `scripts/`:

```text
care-error-aware-evidential-router-v1
care-evidential-conflict-router-v1
care-multiview-conformal-router-source-v1
care-malcertain-router-future-v1
care-rolegraph-conditioned-v4-script
care-rgcondattn-dedup-bounded-source-v1
care-rgcondattn-dedup-bounded-future-v1
```

## Run

```bash
bash scripts/care-error-aware-evidential-router-v1/run_on_server.sh \
  2>&1 | tee artifacts/care-error-aware-evidential-router-v1.console.log
```

The launcher prefers `/home/boyuanjie/envs/tif/bin/python`, then `python3`, then
`python`.  With the established server environment it should print
`Evidential device: cuda`.

## Outputs

```text
artifacts/care-error-aware-evidential-router-source-v1/
artifacts/care-error-aware-evidential-router-future-v1/
```

Package results with:

```bash
zip -r care-error-aware-evidential-router-results.zip \
  artifacts/care-error-aware-evidential-router-source-v1 \
  artifacts/care-error-aware-evidential-router-future-v1 \
  artifacts/care-error-aware-evidential-router-v1.console.log
```
