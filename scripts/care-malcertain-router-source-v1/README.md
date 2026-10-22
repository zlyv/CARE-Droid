# MalCertain-inspired reliability router (source v1)

This experiment changes only the Stage-I-to-Stage-II routing decision. The
existing `Global24` classifier, local graph expert, bounded residual correction,
fixed 2017 folds, labels, and 25% analysis budget remain unchanged.

## What is tested

Five cheap predictions are produced inside each outer fold:

1. all 24 Global24 features;
2. Manifest features;
3. code and call-scale features;
4. sensitive-behaviour features;
5. graph-statistics features.

The outer-training probabilities are strictly cross-fitted. Their disagreement
is converted into MalCertain-style uncertainty descriptors, and an RBF-SVM is
trained to predict whether the complete-view classifier is likely to be wrong.
The 25% highest-risk validation APKs enter Stage II.

Four routes are compared under the same frozen base and expert:

- `A25`: the existing distance-to-0.5 route;
- `ViewRange25`: a simple max-minus-min disagreement route;
- `MVUQ9_SVM25`: base probability plus the nine metrics described in the paper;
- `MVUQ7_SVM25`: base probability plus the seven metric types implemented in
  MalCertain's public uncertainty extractor.

`MVUQ` means multi-view uncertainty. This naming is deliberate: the experiment
borrows MalCertain's uncertainty-feature and correction-model idea, but is not
an exact reproduction of its Bayesian and ensemble prediction procedures.

## Run

Place this directory under `scripts/` on the experiment server, then run from
the repository root:

```bash
bash scripts/care-malcertain-router-source-v1/run_on_server.sh \
  2>&1 | tee artifacts/care-malcertain-router-source-v1.console.log
```

With explicit inputs:

```bash
python scripts/care-malcertain-router-source-v1/run_care_malcertain_router_source_v1.py \
  --features artifacts/p1-features/features.csv \
  --schema artifacts/p1-features/feature_schema.json \
  --p72-oof artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv \
  --expert-predictions artifacts/care-rgcondattn-dedup-bounded-source-v1/source_predictions.csv \
  --budget 0.25
```

Run the artifact-independent check first:

```bash
python scripts/care-malcertain-router-source-v1/selftest_malcertain_router.py
```

No future-year rows are loaded. The SVM reliability target is built only from
cross-fitted errors within the current outer-training fold, so the
outer-validation label never participates in routing.

## Outputs

- `routing_summary.csv`: error recall, routing precision, and accepted risk;
- `source_metrics.csv`: final BA, AUC, F1, fixed/broken/net after routing;
- `source_per_fold_routing.csv`: routing stability across five folds;
- `source_per_fold_metrics.csv`: final metric stability across folds;
- `source_predictions.csv`: view probabilities, uncertainty features, scores,
  route flags, and final probabilities for every APK;
- `fold_audit.csv`: training error counts and fold-level consistency checks;
- `architecture.json`: experimental boundary and configuration;
- `REPORT.txt`: compact result summary.

## Decision rule

At the same 25% budget, focus on `MVUQ9_SVM25` versus `A25`. The new route is
useful only if it captures more first-layer errors, reduces the error rate among
directly accepted APKs, and does not reduce final balanced accuracy. Compare
`MVUQ9_SVM25` with `ViewRange25` to determine whether the learned SVM adds value
beyond a simple disagreement score.
