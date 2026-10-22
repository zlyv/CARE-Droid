# CARE-Droid Committee Decision-Instability Router v1

This is a route-only, equal-budget audit that preserves the previous committee
experiment. It does not retrain LR, RF, XGBoost, SVM, their Platt calibrators,
the official Global24 Stage I, or the frozen bounded local expert.

For frozen Stage-I decision `yhat` and calibrated committee probability `p_m`:

```text
s_m  = yhat*p_m + (1-yhat)*(1-p_m)
S    = mean_m(s_m)
U_DI = 1-S
```

Thus `U_DI` is the committee probability mass assigned to the class opposite
the official Stage-I hard decision:

```text
Stage I predicts malware: U_DI = 1-p_bar
Stage I predicts benign:  U_DI = p_bar
```

This score deliberately does not add PE, MI, or A25. It captures both shared
hesitation and collective/partial opposition to Stage I, but it should be
described as *decision instability*, not as pure epistemic uncertainty: after
averaging, it depends on `p_bar` and the Stage-I decision, not on committee
variance by itself.

## Inputs

The completed v1 experiment must exist:

```text
artifacts/care-committee-uncertainty-router-source-v1/source_sample_scores.csv
artifacts/care-committee-uncertainty-router-future-v1/future_sample_scores.csv
```

These files already contain strict source OOF and 2017-only calibrated future
member probabilities. Reusing them guarantees that only the route score
changes.

## Run

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o care-committee-decision-instability-router-v1.zip -d scripts/
bash scripts/care-committee-decision-instability-router-v1/run_on_server.sh \
  2>&1 | tee artifacts/care-committee-decision-instability-router-v1.console.log
```

## Outputs

```text
artifacts/care-committee-decision-instability-router-source-v1/
  summary.txt
  source_router_metrics.csv
  source_route_overlap.csv
  source_sample_scores.csv
  protocol.json

artifacts/care-committee-decision-instability-router-future-v1/
  summary.txt
  future_annual_metrics.csv
  future_route_overlap.csv
  future_sample_scores.csv
  protocol.json
```

Source routes use exact Top 25% separately in each fixed outer fold. Future
routes use exact Top 25% separately by year. A25, PE, MI, and DI all use the
same frozen BoundedCorrection. Future results are post-hoc diagnostics only and
must not be used to tune the score.
