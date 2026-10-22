# CARE-Droid DI25 No-Ego Need-Gate Probe v1

This package is a new, isolated diagnostic experiment. It does not modify old CARE-Droid code or artifacts, rebuild DI25, reanalyse APKs, consume Ego tensors, or output a replacement CARE-Droid prediction.

## Question

On the already fixed `route_DI == 1` cohort, how readable is the binary target

`need_correct = ((p_stage1 >= 0.5) != label)`

from a true Host-only Stage-II representation?

## Fixed arms

- `S0_STAGE1_LOGIT`: one-dimensional Stage-I logit.
- `S1_GLOBAL24_LOGIT`: fold-local Global24 plus Stage-I logit.
- `S2_NOEGO_HOST_ANOMALY`: 88-D Host-only structural vector: graph token 32, critical Host 24, attended Host 24, scalar summary 8.
- `S3_NOEGO_HOST_ANOMALY_LOGIT`: S2 plus Stage-I logit.

S2/S3 use the historical `DedupBoundedRGCondAttn` Host-only architecture. No Ego file key, encoder, aggregation, gate, zero placeholder, or HostEgo model is used. The delta head exists in the historical training model but its output is neither consumed nor reported by this probe.

## Historical implementation provenance

- Fixed cohort, `route_DI`, existing year/fold, fold-local Global24 and inner-OOF Stage-I probabilities: `care-route-stage2-within-year-matrix-v1/run_within_year_arm.py` (`load_data`, `fold_global24`, `inner_oof_base`).
- True Host-only bags: `care-route-stage2-within-year-matrix-v1/run_future_arm.py::load_bags`, which selects only `host_values`, `host_offsets`, and `real_host_count`.
- Host-only Anomaly model and training: `run_care_rgcondattn_dedup_bounded_source_v1.py::DedupBoundedRGCondAttn` plus the historical v4 `select_epochs`, graph-scaling and `train_epochs` helpers.
- Exact 88-D structural layout: confirmed against `care-di25-host-input-ablation-v1/masked_host_model.py` (`STRUCTURE_ONLY` removes Global24 and Stage-I logit from the 113-D correction input).
- Probe: `care-di25-hostego-feature-sufficiency-v2` / R4 component audit convention, namely `StandardScaler` plus class-balanced `LogisticRegression(C=1, solver=liblinear, max_iter=5000)`.
- Statistics: historical paired sample bootstrap with 10,000 repetitions and seed `20260920`.

No S4 is included: after true Ego removal it would duplicate either the full 113-D correction input (which mixes Global24/logit) or S3, so it has no distinct, clean sufficiency interpretation.

## Leakage boundary

For every year and existing outer fold:

- Stage-II scaling, anomaly prototypes, graph scaling, early stopping and neural fitting use only same-year outer-train APKs.
- Probe `StandardScaler` and class-balanced L2 logistic regression use only routed same-year outer-train APKs.
- Outer-validation APKs are inference-only. Their labels are opened only for the predeclared target and final metrics.
- Existing year/fold assignments are reused exactly; no split is rebuilt.

The Stage-II representation is fitted on all same-year outer-train APKs, matching the historical Stage-II protocol; the need-gate probe itself is fitted only on DI-routed outer-train APKs.

## Hard blockers

Missing inputs, changed schemas/dimensions, duplicate SHA values, incomplete folds, a one-class routed training target, non-finite values, or an incomplete OOF assignment cause `BLOCKED_BY_CONTRACT`. The scripts never silently drop a row, fill a missing feature with zeros, switch to Ego, rebuild routing, or extract APKs.

## Outputs

Under `artifacts/care-di25-noego-needgate-probe-v1/`:

- `cohort_audit.json`
- `noego_contract.json`
- `protocol.json`
- `oof_predictions.csv`
- `fold_audit.csv`
- `representation_training_audit.csv`
- `metrics_by_year.csv`
- `metrics_pooled.csv`
- `bootstrap_comparisons.csv`
- `summary.md`

Bootstrap comparisons are paired S0-vs-S2 and S0-vs-S3 for ROC-AUC and PR-AUC, by year and pooled, with 10,000 fixed-seed resamples.

`oof_predictions.csv` is one row per DI-routed APK and contains `sha256, year, fold, label, p_stage1, stage1_logit, stage1_pred, need_correct, route_DI`, followed by `score_S0..score_S3` and `pred_S0..pred_S3`.

## Server execution

The zip is expected to be uploaded to `/home/boyuanjie/projects/`, one level above the repository:

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-di25-noego-needgate-probe-v1.zip -d scripts/
bash scripts/care-di25-noego-needgate-probe-v1/launch_tmux.sh
tail -F artifacts/care-di25-noego-needgate-probe-v1.console.log
```

To inspect completion:

```bash
tmux capture-pane -p -S -100 -t care-noego-needgate-v1
cat artifacts/care-di25-noego-needgate-probe-v1/summary.md
```

This source package is code-generated only. It has not been executed against the server artifacts.
