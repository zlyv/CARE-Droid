# CARE-Droid frozen future route/Stage-II matrix v1

This package evaluates the predeclared 2 x 2 x 2 matrix on the frozen
2018--2021 diagnostic cohort:

- routers: A25 and DI25, both equal-budget YearTop25;
- Stage-II representations: AnomalyBaseline and MalBenContrast;
- decision heads: independent DirectStageII and FixKeep BoundedCorrection.

DirectStageII predicts an absolute malware probability without Global24 or
`p_stage1` in its expert forward path. On a routed APK it replaces the Stage-I
probability. BoundedCorrection receives the frozen Stage-I state and applies
the source-selected logit cap `DeltaMax=2.25`. Unrouted APKs always retain
Global24.

Every neural model, scaler, prototype, epoch choice, and final L2 classifier is
fit on the fixed 799-APK 2017 source cohort only. Before preflight, the runner
creates the one previously missing artifact: Compact76 features for the 1989
future APKs. This is a lightweight, append-resumable static/API pass stored on
`/mnt/HDD1`; it does not rebuild the completed Host/Ego/local graph caches.
Subsequent runs reuse it after a strict integrity gate. Preflight materializes an
exact 2788-row compact-feature copy: 2017 training labels are retained, while
every label-bearing field is blanked for all 2018--2021 rows. The four expert
jobs consume that copy plus a label-free future router-input file and hard-fail
if any future label is visible. Future labels are read only by the final
summarizer after all four probability files exist. The already extracted Host
and graph assets are reused. The only APK reparse is this one-time future
Compact76 pass, because those rows were never persisted by the earlier
local-asset extraction.

The package is self-contained at the code level: the established CARE v4,
AnomalyBaseline, MalBenContrast, direct Stage-II and committee helper sources
are vendored inside it. It does not require the old experiment-script folders.
If old A25/DI score CSVs are absent, they are not guessed or copied: Global24
and the calibrated LR/RF/XGBoost/SVM committee are deterministically rebuilt
from the frozen 2017 folds. A hard gate requires the reconstructed 2017
Global24 BA to equal the established 0.8196741854636592 before future work can
start. Future labels remain sealed during that reconstruction.

## Server run

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o care-route-stage2-future-matrix-v1.zip -d scripts/
bash scripts/care-route-stage2-future-matrix-v1/launch_tmux.sh
tail -F artifacts/care-route-stage2-future-matrix-v1.console.log
```

By default the future compact cache is written to:

```text
/mnt/HDD1/boyuanjie/tasa-droid-storage/care-route-stage2-future-matrix-v1/compact_future_2018_2021_fixed3
```

It hard-fails on duplicates, failures, missing rows, outside-lineage rows, or
anything other than the frozen 793/399/397/400 annual counts. A clean
interrupted extraction resumes from existing unique rows. To choose another
large-disk directory, set:

```bash
export CARE_FUTURE_COMPACT_ROOT=/mnt/HDD1/.../another_clean_directory
```

Source/future feature and schema paths can also be overridden with
`CARE_SOURCE_FEATURES`, `CARE_SOURCE_SCHEMA`, `CARE_FUTURE_FEATURES`, and
`CARE_FUTURE_SCHEMA`.

The launcher deliberately supports two existing Python environments. Neural
Stage-II training defaults to `/home/boyuanjie/envs/tif/bin/python` so CUDA is
retained. Router reconstruction auto-selects the first interpreter containing
NumPy, scikit-learn, PyTorch and XGBoost, preferring
`/home/boyuanjie/venvs/malscan/bin/python`. Explicit overrides are:

```bash
export ML_PYTHON_BIN=/home/boyuanjie/envs/tif/bin/python
export ROUTER_PYTHON_BIN=/path/to/python/with/xgboost
```

Primary outputs:

```text
artifacts/care-route-stage2-future-matrix-v1/results/REPORT.txt
artifacts/care-route-stage2-future-matrix-v1/results/future_annual_metrics.csv
artifacts/care-route-stage2-future-matrix-v1/results/future_pooled_metrics.csv
artifacts/care-route-stage2-future-matrix-v1/results/future_macro_mean_metrics.csv
artifacts/care-route-stage2-future-matrix-v1/results/future_pairwise.csv
artifacts/care-route-stage2-future-matrix-v1/results/future_route_overlap.csv
artifacts/care-route-stage2-future-matrix-v1/results/future_sample_predictions.csv
```

These years have already been inspected, so all results are diagnostic-only.
They must not be used to revise the router, budget, bound, representation, or
decision head.
