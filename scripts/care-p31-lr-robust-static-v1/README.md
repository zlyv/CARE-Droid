# CARE-Droid P31-LR robust static representation v1

This experiment keeps logistic regression, calibration, threshold 0.5, fixed folds,
MeanOpposition Top25%, A0 Role C43-RQ and max_delta=2.25 unchanged.  It changes
only a label-free transform of the already-normalized 31-dimensional P31
composition.

Arms:

- `L0_RAW`: verified current raw P31-LR.
- `L1_HELLINGER`: square-root/Hellinger composition.
- `L2_LOG`: log composition with frozen pseudocount 1e-4.
- `L3_PRESENCE`: Role presence only.
- `L4_RAW_PRESENCE`: raw composition plus Role presence.
- `L5_CLR_PRESENCE`: centered log-ratio plus Role presence.
- `L6_HELLINGER_PRESENCE`: Hellinger composition plus Role presence.

The source ranking is frozen after 2017 fixed-fold OOF and before future labels
are used.  Frozen 2017-to-2018--2021 and independent same-year OOF are diagnostics
only.  Every arm is also evaluated with the identical Stage-II.

Run on GPU 1:

```bash
PLRRS_GPUS=1 bash scripts/care-p31-lr-robust-static-v1/launch_tmux.sh
tail -F outputs/care-p31-lr-robust-static-v1.console.log
```

Completion marker:

```text
CARE_P31_LR_ROBUST_STATIC_PIPELINE_DONE
```
