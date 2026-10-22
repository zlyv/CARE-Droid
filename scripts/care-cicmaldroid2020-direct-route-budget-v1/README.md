# CICMalDroid2020 direct Stage-II route-budget diagnostic

This is an external, chronological diagnostic using the four archives supplied
by the user: Benign, Adware, Banking and Riskware.  The standard SMS cohort is
not present, so the output must not be described as the complete benchmark.

The frozen method is:

1. 2017+2018 calibrated P31-SVM;
2. one SVM-anchored MeanOpposition ranking;
3. nested Top25%, Top35%, Top45% and Top50% routes;
4. one shared U2 direct C43 Stage-II model.

For a routed APK with complete Stage-II assets, the direct Stage-II probability
replaces the Stage-I probability.  There is no bounded residual, tanh
correction, max-delta integration or budget-specific retraining.  Missing
Stage-II assets conservatively retain the Stage-I probability.

The pipeline first inventories the tar archives and checks free disk space.  It
then extracts APKs with traversal protection, content-addresses them by SHA256,
deduplicates them, excludes CARE-Droid 2017--2021 overlaps and extracts the
Top50 union only once for Stage-II.

Riskware is counted as malware in the inclusive primary table and excluded in
a strict secondary table.  Every cohort is also reported separately.  External
labels are sealed until evaluation and cannot select a formal CARE-Droid route
budget.
