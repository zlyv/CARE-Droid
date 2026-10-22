# CARE-Droid D0 correction-score anatomy v1

This source-only package audits the reported D0 FIXED-vs-BROKEN AUC before any calibration is allowed. It keeps P31-RF, RF-anchored MeanOpposition DI25, A0 Role C43-RQ, and `max_delta=2.25` fixed and never loads future years.

The earlier `0.805` is traced to the label-free saved three-seed ensemble diagnostic `abs_bounded_delta`. Because raw seed-level deltas were not saved, `raw_delta` is explicitly marked unavailable rather than reconstructed incorrectly.

The locked evidence already indicates only 2 of 4 eligible folds share the pooled direction. The package independently reconstructs the complete anatomy, fold AUCs, distribution statistics and strength curve, then stops before M1-M3 if that instability is reproduced.

```bash
bash scripts/care-d0-correction-score-calibration-v1/launch_tmux.sh
tail -F artifacts/care-d0-correction-score-calibration-v1.console.log
```
