# CARE-Droid P31 classifier family + Stage-II v1

This experiment answers a different question from the earlier Stage-I-only
classifier comparison.  Every arm is a complete two-layer system:

`calibrated P31 arm -> arm-anchored MeanOpposition Top25% -> A0 Role C43-RQ -> bounded correction (max_delta=2.25)`

Arms: `LR`, `RF`, `XGB`, `SVM`, and `MEAN4` (control).  The formal RF arm is
read-only and reused from the frozen artifacts.  The four other arms retrain an
identical-capacity Stage-II using arm-specific nested OOF Stage-I probabilities.

The primary comparison is strict 2017 fixed-fold OOF.  Frozen 2017-to-2018--2021
and independent same-year five-fold runs are diagnostics and do not change the
formal RF method.

Run on the server (GPU 1 by default):

```bash
bash scripts/care-p31-classifier-family-stage2-v1/launch_tmux.sh
tail -F outputs/care-p31-classifier-family-stage2-v1.console.log
```

If both GPUs are confirmed idle, launch with:

```bash
PCFS2_GPUS=0,1 bash scripts/care-p31-classifier-family-stage2-v1/launch_tmux.sh
```

The final table contains Stage-I BA, Stage-I+Stage-II BA, gain in percentage
points, fixed, broken, net, and class-side breakdowns for every arm and scope.
