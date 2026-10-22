# CARE-Droid no-Ego S3 gate-safety source diagnostic v1

This read-only 2017 diagnostic asks whether a safer operating point can retain
S3's additional fixes while reducing broken samples. It does not retrain any
model, consume Ego, inspect a future row, or search a threshold on the target
fold.

For each target fold, a threshold in `[0.50,1.00]` with step `0.01` is selected
on the other four folds using the predeclared utility:

`Fixed - 1.5 * Broken`

Ties prefer fewer broken, then more fixed, then the higher threshold. Because
the historical OOF predictors for the four selection folds may have seen rows
from the target fold during their own training, the result is explicitly a
development cross-fold diagnostic, not a fully nested unbiased estimate.

Upload the zip to `/home/boyuanjie/projects/`, then:

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-di25-noego-gate-safety-source-v1.zip -d scripts/
bash scripts/care-di25-noego-gate-safety-source-v1/run_on_server.sh \
  2>&1 | tee artifacts/care-di25-noego-gate-safety-source-v1.console.log
cat artifacts/care-di25-noego-gate-safety-source-v1/REPORT.txt
```

This is short and does not require tmux.
