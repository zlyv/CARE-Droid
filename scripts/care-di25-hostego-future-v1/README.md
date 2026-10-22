# CARE-Droid frozen DI25 Host/Ego future diagnostic v1

This package performs the post-selection temporal diagnostic for the frozen
source-selected configuration:

- router: DI25 / per-year Top-25% using the already frozen, label-free scores;
- local representation: directed Host + one-hop Host Ego (M1);
- evidence: Anomaly;
- decision head: logit-space bounded correction;
- bound: `max_delta = 2.25`;
- source training cohort: the fixed 799 APKs from 2017 only.

The package does **not** rebuild DI25, retrain M0, change M0/M1 source
predictions, or use future labels during training, routing, epoch selection, or
inference. It reuses the completed M0 future probabilities. It trains only the
deployment M1 ensemble on all 799 source APKs, freezes 1,989 label-free future
probabilities, and only then lets the final summarizer read labels.

## Run

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o care-di25-hostego-future-v1.zip -d scripts/
bash scripts/care-di25-hostego-future-v1/launch_tmux.sh
tail -F artifacts/care-di25-hostego-future-v1.console.log
```

The launcher is resumable. If all 1,989 M1 probabilities and the matching
source-only protocol already pass their contract, training is skipped and only
the deterministic final summary is regenerated.

## Outputs

- `preflight.json`: source-result, route, cohort, asset, and label-seal gates;
- `m1_hostego_anomaly/future_m1_predictions_label_free.csv`: frozen M1 outputs;
- `m1_hostego_anomaly/protocol.json`: source-only training protocol;
- `results/future_annual_metrics.csv`: 2018--2021 annual results;
- `results/future_pooled_metrics.csv`: pooled temporal diagnostic;
- `results/future_M0_M1_paired.csv`: exact paired M0/M1 McNemar audit;
- `results/future_sample_predictions.csv`: aligned per-APK audit table;
- `results/future_ego_usage.csv`: annual Ego-use accounting;
- `results/unseal_audit.json`: proof that label access occurs after prediction freeze;
- `results/REPORT.txt`: concise report.

`DirectSecondary` is a parameter-free diagnostic from the same correction
head. It is not a separately trained model and is not the primary method.

All 2018--2021 results are diagnostic-only. They must not be used to revise the
router, representation, evidence design, bound, or decision head.
