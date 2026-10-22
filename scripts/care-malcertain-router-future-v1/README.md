# MalCertain-inspired reliability router: future diagnostic v1

This package evaluates whether the multi-view uncertainty router learned from
the fixed 799-APK 2017 cohort transfers to 2018--2021.

The test keeps the following components frozen:

- Global24 feature definition and deployment probability;
- five semantic views and the SVM router are trained with 2017 labels only;
- the deduplicated bounded RGCondAttn expert is reused from its existing future
  prediction artifact and is not retrained here;
- the final threshold remains 0.5.

The reused `A25__score` is the percentile-calibrated Global24 ambiguity from
the established future artifact.  The original pipeline first computes the
raw ambiguity `1 - 2*abs(p-0.5)` and then maps it against a cross-fitted 2017
reference distribution.  The calibrated score remains the routing input; the
raw ambiguity is exported separately as `A25__raw_ambiguity` for auditing.

Future labels are read only after all uncertainty scores, route flags, and
final probabilities are fixed.  Because 2018--2021 have already been inspected
elsewhere in the project, the results are diagnostic-only.

## Primary and secondary protocols

`YearTop25` is the primary equal-budget comparison.  It selects the highest
risk 25 percent independently within every future year, using no labels.

`SourceQ75` is a stricter score-transport audit.  It applies the 75th-percentile
threshold fitted on 2017 scores directly to every future year, so its future
route ratio is allowed to drift.

The principal comparison is:

- `A25_YearTop25` versus `MVUQ9_SVM_YearTop25`.

`MVUQ7_SVM_YearTop25` is the metric-family sensitivity check and
`ViewRange_YearTop25` tests whether simple disagreement is sufficient.

## Dependencies

The following directories are expected under `scripts/`:

- `care-malcertain-router-source-v1`;
- `care-multiview-conformal-router-source-v1`;
- established CARE source/future dependency scripts;
- `care-rgcondattn-dedup-bounded-future-v1` must already have produced:
  `artifacts/care-rgcondattn-dedup-bounded-future-diagnostic-v1/future_predictions_diagnostic.csv`.

## Run

From the repository root:

```bash
bash scripts/care-malcertain-router-future-v1/run_on_server.sh \
  2>&1 | tee artifacts/care-malcertain-router-future-v1.console.log
```

## Outputs

- `future_annual_routing_diagnostic.csv`: annual error capture and accepted risk;
- `future_annual_metrics_diagnostic.csv`: annual BA/AUC/F1 and fixed/broken/net;
- `future_pooled_routing_diagnostic.csv`: pooled routing quality;
- `future_pooled_metrics_diagnostic.csv`: pooled end-to-end metrics;
- `future_predictions_diagnostic.csv`: per-APK views, uncertainty features,
  route flags, and final probabilities;
- `audit.json`: leakage boundary, thresholds, feature dimensions, and route counts;
- `REPORT.txt`: compact console-readable summary.
