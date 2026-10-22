# CARE-Droid independent within-year matrix v1

This package repeats the frozen 2(router) x 2(Stage-II representation) x
2(decision-head) matrix independently inside 2018, 2019, 2020 and 2021.
Each year receives a shuffled stratified five-fold outer OOF evaluation. Every
outer-validation APK is predicted by models, scalers and prototypes fitted only
on the other four folds from the same year.

The experiment is a same-distribution diagnostic. It deliberately uses labels
from each evaluated year for within-year training and therefore must never be
reported as temporal, future-year, zero-day or deployment performance. Its
purpose is to distinguish representation/model capacity from the degradation
caused by training on 2017 and transporting to later distributions.

Frozen design choices:

- A25 and DI25 route exactly 25% of each annual cohort using OOF scores;
- Global24 and all four committee members are trained inside each outer fold;
- DirectStageII replaces the OOF Global24 probability only for routed APKs;
- BoundedCorrection uses inner-OOF Stage-I probabilities on outer-train and
  keeps the source-selected logit cap `DeltaMax=2.25`;
- three neural seeds remain `900000,900001,900002`;
- no budget, bound or architecture is selected from 2018--2021 results.

The completed Compact76 and Host/Ego assets are reused. No APK extraction is
repeated. The run is substantially longer than the frozen-future evaluation:
four expert arms each train 4 years x 5 outer folds x 3 seeds. It runs in tmux
and writes an atomic checkpoint after every completed outer fold, so an
interrupted run resumes without repeating completed folds.

## Run

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o care-route-stage2-within-year-matrix-v1-fixed1.zip -d scripts/
bash scripts/care-route-stage2-within-year-matrix-v1/launch_tmux.sh
tail -F artifacts/care-route-stage2-within-year-matrix-v1.console.log
```

Primary report:

```text
artifacts/care-route-stage2-within-year-matrix-v1/results/REPORT.txt
```

Tabular outputs include annual, pooled, macro-mean and sample-level OOF
predictions. Annual and macro metrics are the primary interpretation because
each year is fitted independently.
# Compact76 feature-space contract

The Compact feature dictionary is frozen from the 799-APK 2017 source cohort.
Fields first observed in 2018--2021 (for example
`role_node__package_installation` and `role_edge__package_installation`) are
recorded in `compact76_contract_audit.json` and excluded from every model.
Ratio normalization is re-fitted on each outer-training fold (and on each
inner-training fold for nested Stage-I OOF predictions); validation rows never
fit preprocessing statistics.
