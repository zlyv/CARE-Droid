# CARE-Droid Formal Diagnostics v1

Post-freeze diagnostics for the immutable formal method:

`P31-RF -> RF-anchored MeanOpposition DI25 -> A0 Role C43-RQ -> bounded correction (2.25)`.

The four P31 classifiers are used only by the router. Stage-I prediction is the calibrated Random Forest alone.

Two protocols are produced:

1. Frozen 2017 training to 2018--2021 temporal diagnostics. Future labels are joined only by `summarize.py` after label-free predictions are saved.
2. Independent same-year fixed five-fold OOF for 2018--2021. This uses each year's labels for training and is not temporal generalization.

The package also reports Full100 as a diagnostic work-budget control. It never changes the formal DI25 method.

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-droid-formal-diagnostics-v1-fixed4.zip -d scripts/
bash scripts/care-droid-formal-diagnostics-v1/launch_tmux.sh
tail -F artifacts/care-droid-formal-diagnostics-v1.console.log
```

If an older GPU task is still active, the pipeline prints `WAITING_FOR_GPU*_IDLE` and starts automatically when that GPU becomes idle.

The future InterZ input is resolved by content rather than a fixed historical
directory.  An existing cache is reused only when its ordered 1,989 SHA list,
961-dimensional arrays, and frozen null-model parameters all match.  If no
such cache exists, the pipeline builds the same label-free InterZ from the
already extracted endpoint cache and checkpoints it under this experiment;
this does not reanalyse APKs or load future labels.

Fixed folds are discovered from prior same-year OOF assets by content.  An
asset is accepted only when its SHA rows exactly cover the frozen 1,989-APK
cohort (disjoint year-group files from the same experiment may be combined).
Its fold IDs are preserved exactly, while year membership is taken from the
frozen label-free cohort.  Sources and stale year annotations are written to
`fold_year_alignment_audit.json`; the pipeline never invents a fold for a
missing APK.

The endpoint tensor cache is likewise resolved by SHA coverage rather than a
legacy pathname.  A single 2,788-row cache is preferred; if source and future
rows live in separate caches, their already-extracted successful JSONL rows
are merged verbatim into an experiment-local cache.  No APK reanalysis or
feature recomputation occurs during that merge.

Final results:

```bash
cat artifacts/care-droid-formal-diagnostics-v1/REPORT.txt
column -s, -t artifacts/care-droid-formal-diagnostics-v1/aggregate_metrics.csv
column -s, -t artifacts/care-droid-formal-diagnostics-v1/within_year_fold_metrics.csv
cat artifacts/care-droid-formal-diagnostics-v1/final_summary.json
```
