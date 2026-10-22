# CARE-Droid P31-LR source-safe fusion v1

This experiment optimizes only the 2017 Stage-I P31 logistic model. It does not load future rows and does not run Stage-II.

Candidates are fixed label-free P31 bases plus fold-internal regularization selection. The pairwise arm is a bounded second-order control, not an unrestricted deep model. `C0_COUNT_SUMMARY` is a shortcut control and cannot be selected.

Run with `bash scripts/care-p31-lr-source-safe-fusion-v1/launch_tmux.sh` and collect with `bash scripts/care-p31-lr-source-safe-fusion-v1/collect_results.sh`.

