# Manifest selective validation V1

Independent 2017-only repeat of fixed ManifestOnly/RefPresence31 + frozen Direct-C43 at 10% and 25% uncertainty routing. No original files, checkpoints, graph cache or prior results are altered. No new downloaded APKs; no future cohort is read. Calibration and RF are delegated to unchanged stage1_core.

Three fixed seeds; same source outer folds; paired APK uncertainty intervals. Each seed is a separate model, not a three-member deployment ensemble. Frozen C43 probabilities are reused; C43 is not trained or altered. This is exploratory repeated-source evidence, not independent validation of the previously selected 10% budget.

20 source APKs are label-free sampled by fold and APK size quartile. Static-only feature extraction is timed in isolated child processes with two CPU threads, 20 GiB virtual memory cap and 20-minute per-component timeout. Original APK hashes are checked. No installation/execution, page-cache clearing or environment installation. All child outputs are inside the new run.

The graph builders measure compact/endpoint components. Full C43 preprocessing/forward is NOT timed, so end-to-end latency remains UNVERIFIED. Extraction failures and timeouts remain in the results. No performance-based threshold/seed selection or automatic future run.

Run with MSV_OUT pointing to a fresh direct run_* child of outputs/care_manifest_selective_validation_v1. `bash scripts/care-manifest-selective-validation-v1/run.sh --preflight-only` verifies assets/contracts; subsequent run uses the same frozen manifest. `--resume` reuses completed fold predictions and completed timings, but refuses an interrupted cost item rather than treating its partial cache as cold work. No files are overwritten.

Main outputs: protocol.json, source_metrics.csv, source_fold_metrics.csv, source_per_apk.csv, paired_statistics.csv, cost_measurements.csv, cost_summary.csv, baseline_integrity_check.json, final_report.md and final_decision.json.
