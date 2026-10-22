# CARE-Droid frozen DI25 Host/Ego within-year diagnostic v1

This package evaluates the source-selected M1 representation under an
independent same-year protocol. Each of 2018, 2019, 2020 and 2021 is split into
five stratified outer folds. Every validation APK is predicted by models fitted
only on the other four folds from the same year.

Frozen method:

- router: existing strict same-year OOF DI25, Top-25% per year;
- M0 control: existing Host + Anomaly + Bounded predictions, reused verbatim;
- M1: directed Host + one-hop Host Ego + Anomaly + Bounded;
- `max_delta = 2.25` in logit space;
- three neural seeds: `900000,900001,900002`;
- parameter count: 34,613.

For each outer fold, Global24 preprocessing, inner-OOF Stage-I probabilities,
anomaly scaling/prototypes, epoch selection, and Stage-II fitting use outer
training rows only. Outer-validation labels are used only for final scoring.
APK/static graph extraction is not repeated.

These are same-distribution diagnostics. They are **not** future-year,
zero-day, or temporal-generalization results.

## Run

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o care-di25-hostego-within-year-v1.zip -d scripts/
tmux kill-session -t care-di25-hostego-within-year-v1 2>/dev/null || true
bash scripts/care-di25-hostego-within-year-v1/launch_tmux.sh
tail -F artifacts/care-di25-hostego-within-year-v1.console.log
```

The 20 year/fold jobs checkpoint after every fold and resume without repeating
completed folds.

## Main outputs

- `results/REPORT.txt`
- `results/within_year_annual_metrics.csv`
- `results/within_year_pooled_metrics.csv`
- `results/within_year_M0_M1_paired.csv`
- `results/within_year_ego_usage.csv`
- `results/within_vs_temporal_gap.csv` when the frozen temporal result exists.

`DirectSecondary` is a parameter-free diagnostic from the same residual head;
it is not a separately trained Stage-II classifier.
