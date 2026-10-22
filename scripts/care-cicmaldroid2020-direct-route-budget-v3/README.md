# CICMalDroid2020 direct Stage-II route-budget diagnostic

P31 failures are explicitly audited and excluded from every downstream
denominator. They are never represented as zero vectors. See
`p31_exclusion_audit.json`, `p31_failures.json`, and
`p31_success_manifest_label_free.csv`.

This is an external, chronological diagnostic using a deterministic stratified
sample from the three usable archives supplied by the user: Benign, Banking and
Riskware.  The incomplete Adware archive is explicitly skipped, and SMS is not
present.  At most 500 APKs are selected uniformly by reservoir sampling from
each usable archive before feature extraction.  The seed is frozen in
`common.py`.  Output must not be described as the complete benchmark or as a
full-corpus result.

The frozen method is:

1. 2017+2018 calibrated P31-SVM;
2. one SVM-anchored MeanOpposition ranking;
3. nested Top25%, Top35%, Top45% and Top50% routes;
4. one shared U2 direct C43 Stage-II model.

For a routed APK with complete Stage-II assets, the direct Stage-II probability
replaces the Stage-I probability.  There is no bounded residual, tanh
correction, max-delta integration or budget-specific retraining.  Missing
Stage-II assets conservatively retain the Stage-I probability.

The pipeline identifies APKs by ZIP local-file-header magic rather than filename
suffix, scans the usable tar archives, freezes each cohort's random sample
and checks free disk space for the sampled APKs.  It then extracts only selected
APKs with traversal protection, content-addresses them by SHA256, deduplicates
them, excludes CARE-Droid 2017--2021 overlaps and extracts the Top50 union only
once for Stage-II.

Riskware is counted as malware in the inclusive primary table and excluded in
a strict secondary table.  Every cohort is also reported separately.  External
labels are sealed until evaluation and cannot select a formal CARE-Droid route
budget.
