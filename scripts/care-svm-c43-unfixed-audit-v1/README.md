# CARE-Droid SVM+C43 captured-but-unfixed audit v1

This is a fast, read-only, 2017 source-only diagnostic. It does not train a new model, use a GPU, load future years, or change the formal assets.

It separates the 84 routed SVM Stage-I errors into: fixed, correct direction but too weak, wrong direction, near-zero correction, and Stage-I logit margin beyond the frozen `max_delta=2.25`.

The included scale sweep is descriptive only. A larger scale is not accepted as a method. It only decides whether a separate strict nested-OOF calibration experiment is scientifically justified.

Run:

```bash
bash scripts/care-svm-c43-unfixed-audit-v1/launch_tmux.sh
tail -F outputs/care-svm-c43-unfixed-audit-v1.console.log
```

Collect:

```bash
bash scripts/care-svm-c43-unfixed-audit-v1/collect_results.sh
```
