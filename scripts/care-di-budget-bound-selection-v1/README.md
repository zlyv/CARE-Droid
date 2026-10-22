# CARE-Droid DI budget and bounded-correction selection v1

This package fills two missing source-side experiments:

1. whether the established 25% Stage-II budget is a reasonable operating
   point after replacing A25 with decision instability (DI);
2. whether the bounded logit-correction limit `max_delta=1.5` is supported by
   2017-only validation.

The package performs **exact retraining**, not post-hoc scaling.  For every
candidate `max_delta`, the deduplicated RGCondAttn correction head is trained
with seeds 0,1,2 under the fixed 2017 outer folds.  A25 and DI are then compared
over the same budget grid.

The established dedup source wrapper historically wrote `1.5` into metadata
even when `--max-delta` was overridden. This package corrects that metadata
after each completed run; it does not alter predictions.

## Default grid

```text
max_delta (logit): 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0
Stage-II budget:   5%, 10%, 15%, 20%, 25%, 30%, 35%, 40%, 50%
```

`max_delta` is in **logit space**:

```text
z_final = z_stage1 + max_delta * tanh(o)
p_final = sigmoid(z_final)
```

It is not an additive probability bound.

## Scientific protocol

- Source model comparison uses the fixed 2017 outer folds only.
- The main deployment rule declares a budget ceiling of 25% before selection.
- Within that ceiling, selection maximizes OOF balanced accuracy, then net
  repairs, then fewer broken samples, then the smaller budget and bound.
- A leave-one-outer-fold-out selection audit chooses budget/bound on the other
  folds and evaluates the held-out fold.  This is the preferred source estimate
  after parameter selection.
- The pooled source-selected row is a development decision and should not be
  presented as an untouched test estimate.
- Only after `selected_source_config.json` is frozen is the selected model
  trained on all 2017 data and diagnosed on 2018-2021.
- Future diagnostics report both an exact per-batch quota (`YearTopB`) and a
  fixed 2017 score threshold (`SourceThreshold`).  The latter exposes route
  ratio drift without using future labels to choose routes.

## Dependencies already expected on the server

The following completed packages/artifacts are reused:

```text
scripts/care-rolegraph-conditioned-v4-script/
scripts/care-rolegraph-conditioned-v4-future-script/
scripts/care-rgcondattn-dedup-bounded-source-v1/
scripts/care-rgcondattn-dedup-bounded-future-v1/
artifacts/care-committee-uncertainty-router-source-v1/source_sample_scores.csv
artifacts/care-committee-uncertainty-router-future-v1/future_sample_scores.csv
```

## Run source selection first

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o care-di-budget-bound-selection-v1.zip -d scripts/

bash scripts/care-di-budget-bound-selection-v1/run_source_grid_on_server.sh \
  2>&1 | tee artifacts/care-di-budget-bound-selection-v1-source.console.log
```

This is a relatively expensive seven-bound, three-seed source run.  To run a
smaller preliminary grid without changing the code:

```bash
CAPS=0.5,1.0,1.5,2.0 \
bash scripts/care-di-budget-bound-selection-v1/run_source_grid_on_server.sh \
  2>&1 | tee artifacts/care-di-budget-bound-selection-v1-source-quick.console.log
```

Do not run the future stage until the source report has been inspected and the
JSON has been frozen.

## Run frozen future diagnostic

```bash
bash scripts/care-di-budget-bound-selection-v1/run_selected_future_on_server.sh \
  2>&1 | tee artifacts/care-di-budget-bound-selection-v1-future.console.log
```

## Main outputs

```text
artifacts/care-di-budget-bound-selection-v1/source_selection/
  joint_grid_metrics.csv
  joint_grid_per_fold.csv
  crossfit_selection_choices.csv
  crossfit_selection_metrics.csv
  bootstrap_intervals.csv
  source_selected_predictions.csv
  selected_source_config.json
  REPORT.txt

artifacts/care-di-budget-bound-selection-v1/future_diagnostic/
  future_annual_metrics.csv
  future_pooled_metrics.csv
  future_sample_predictions.csv
  protocol.json
  REPORT.txt
```

## Reporting guidance

Use the experiment to distinguish two claims:

- If 25% lies near the BA-cost Pareto knee, describe it as a source-supported
  efficiency/effectiveness operating point.
- If a smaller or larger budget is consistently selected, do not retain 25%
  merely because it was used previously; report 25% as an equal-budget baseline
  and freeze the newly selected source-only operating point for deployment.

Never revise budget or `max_delta` after inspecting 2018-2021 results.
