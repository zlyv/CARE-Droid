# Frozen Manifest cascade: 2018–2021

Train ManifestOnly and the existing RefPresence31 reference using original 2017 data only (799 APKs). Fixed RF300, training-only 3fold Platt calibration, seeds 20261018/19/20. Verify saved-model prediction equivalence with the historical stage1_core helper on synthetic data. Preserve 10% and 25% per-year uncertainty routing and threshold 0.5. Reuse all-APK HDBC H0 frozen SCBE M0 T1 probabilities, never rolling checkpoints. No APK execution, extraction, new data, C43 training or environment installation.

Feature input definitions and hyperparameters unchanged from Manifest validation. Future features are read without labels during prediction. All six first-stage models and predictions are frozen before evaluation reads expert predictions/labels. Historical outputs are read-only; hash before and after. All generated files go into a fresh direct run_* child of outputs/care_manifest_frozen_future_v1 selected by MFF_OUT.

Outputs: future_yearly_metrics.csv (every seed and year), future_per_apk_predictions.csv, future_pooled_metrics.csv, future_macro_metrics.csv, training_audit.csv, checkpoints/, models_frozen_before_evaluation.json, baseline_integrity_check.json, final_report.md, final_decision.json. No automatic selection, tuning, seed ensembling, within-year training or follow-up. Historical development analysis is not a new independent blind test; no end-to-end cost claim.

Run `bash scripts/care-manifest-frozen-future-v1/run.sh --preflight-only` with MFF_OUT then launch the same script in tmux with --resume. Completed checkpoints may be reused; completed final runs cannot be overwritten. If interrupted while writing report tables, preserve partial results and use a fresh run directory rather than overwrite.
