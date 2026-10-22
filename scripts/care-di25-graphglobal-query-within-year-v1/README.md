# CARE-Droid N0/Q independent within-year OOF v1

This diagnostic compares `N0_NO_GRAPHGLOBAL` and `Q_QUERY_ONLY` independently
within 2018, 2019, 2020, and 2021. Every year uses its existing strict five-fold
OOF Stage-I probability and DI25 route. For each outer fold, inner-OOF Stage-I
probabilities, anomaly prototypes, scaling, epoch selection, and Stage-II fitting
use only the other four folds from that same year.

Both arms are Host-only, have 33,036 parameters, use the same 113-D bounded head
and `max_delta=2.25`, and do not load any Ego tensor. N0 runs on GPU 0 and Q on
GPU 1. Twenty fold checkpoints per arm make the experiment resumable.

This is same-year development/diagnostic evidence. It is not temporal
generalization, unseen-future, zero-day, or final-holdout evidence.

Upload the ZIP to `/home/boyuanjie/projects/`, then run:

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-di25-graphglobal-query-within-year-v1.zip -d scripts/
bash scripts/care-di25-graphglobal-query-within-year-v1/launch_tmux.sh
tail -F artifacts/care-di25-graphglobal-query-within-year-v1.console.log
```

Per-arm logs:

```bash
tail -F artifacts/care-di25-graphglobal-query-within-year-v1/logs/N0_NO_GRAPHGLOBAL.log
tail -F artifacts/care-di25-graphglobal-query-within-year-v1/logs/Q_QUERY_ONLY.log
```

Final report:

`artifacts/care-di25-graphglobal-query-within-year-v1/results/REPORT.txt`
