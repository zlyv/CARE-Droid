# CARE-Droid P31 calibrated classifier-family comparison v1

This experiment compares LR, RF, XGB and RBF-SVM as independent calibrated Stage-I classifiers on the same P31 representation. MEAN4 is retained as a control.

Protocols:

1. 2017 fixed-fold OOF: primary source comparison.
2. Frozen 2017 training to 2018–2021: diagnostic only.
3. Independent same-year fixed five-fold OOF: distribution-matched diagnostic, not temporal generalization.

No Stage-II output enters the comparison. The formal RF is not automatically changed, and future results are never used for selection.

```bash
bash scripts/care-p31-classifier-family-v1/launch_tmux.sh
tail -F outputs/care-p31-classifier-family-v1.console.log
bash scripts/care-p31-classifier-family-v1/collect_results.sh
```

