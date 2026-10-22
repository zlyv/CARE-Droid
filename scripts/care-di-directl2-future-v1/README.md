# CARE-Droid DI25 + Direct-L2 future diagnostic v1

This package runs the future diagnostic for the source-side finding that
`DI25 + DirectL2` outperformed `DI25 + BoundedCorrection`.

It trains the independent Direct-L2 RGCondAttn expert on all fixed 2017 source
APKs only, predicts 2018--2021, and then evaluates:

| Router | BoundedCorrection | Direct-L2 |
|---|---|---|
| A25 YearTop25 | evaluated | evaluated |
| DI25 YearTop25 | evaluated | evaluated |

The Direct-L2 head predicts an absolute malware probability from local evidence
and excludes Global24 features and the Stage-I probability/logit. For routed
APKs, this probability replaces the Stage-I probability.

Future years are diagnostic-only because they have already been inspected.
Do not tune the router or the second-layer head using these results.

## Prerequisites

```text
scripts/care-rgcondattn-direct-l2-source-v1/
artifacts/care-committee-decision-instability-router-future-v1/future_sample_scores.csv
```

The second prerequisite is produced by `care-committee-decision-instability-router-v1`.

## Run

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o care-di-directl2-future-v1.zip -d scripts/
bash scripts/care-di-directl2-future-v1/run_on_server.sh \
  2>&1 | tee artifacts/care-di-directl2-future-v1.console.log
```

Output:

```text
artifacts/care-di-directl2-future-diagnostic-v1/
  REPORT.txt
  future_predictions_diagnostic.csv
  future_di_directl2_annual_metrics.csv
  future_di_directl2_pairwise.csv
  future_di_directl2_sample_scores.csv
  protocol.json
```

The primary readout is `DI25_YearTop25__DirectL2` versus
`DI25_YearTop25__BoundedCorrection`.
