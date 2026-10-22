# P31-LR + routing-only committee + P0-ZERO6 C43 complete diagnostics

Runs two distinct protocols without changing the locked source method:

1. Frozen 2017 training to label-free 2018-2021 prediction. Future labels are joined only by the final summarizer.
2. Independent same-year five-fold OOF. Each year rebuilds P31-LR, all four routing members, Global24 expert context, backgrounds, and C43 using only that year's outer-train fold.

Stage-I classification uses calibrated P31 Logistic Regression only. RF/XGB/SVM are trained only to form LR-anchored MeanOpposition routing. Stage-II is the locked P0-ZERO6 C43-RQ scalar bounded head with `max_delta=2.25`; no Provenance6 signal or Ego tensor is consumed.

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-p31lr-c43-p0-complete-diagnostics-v1.zip -d scripts/
bash scripts/care-p31lr-c43-p0-complete-diagnostics-v1/launch_tmux.sh
tail -F artifacts/care-p31lr-c43-p0-complete-diagnostics-v1.console.log
```

Final report: `artifacts/care-p31lr-c43-p0-complete-diagnostics-v1/results/REPORT.txt`.
