# CARE-Droid Global24 Code/Call confirmation source v1

Strict 2017-only confirmation experiment. It compares `FULL24` against removal
of all three Code/Call size variables and against removal of each variable
individually under identical outer folds, inner-OOF calibration, seeds, and L2
logistic-regression protocol.

Every variant within an outer fold uses the identical `protocol_seed`; there is
no variant-specific seed offset. The emitted `timing.csv` records that seed for
every fit so this fairness condition can be audited directly.

The canonical frozen DI25 route is read only to verify identity and to report
routed/unrouted strata. It is never recomputed, blended, or replaced. No future
row is loaded and Stage II is never trained.

Outputs:

- `REPORT.txt`
- `aggregate_metrics.csv`
- `per_fold_metrics.csv`
- `paired_vs_FULL24.csv`
- `route_stratified_metrics.csv`
- `source_oof_predictions.csv`
- `timing.csv`
- `protocol.json`

The strict source simplification rule requires positive paired net, exact
McNemar p <= 0.05, and nonnegative BA in at least four of five folds. Even a
pass does not authorize future evaluation or replacement of the frozen official
Stage-I without human approval.
