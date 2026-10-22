# SVM + Direct-C43 frozen future diagnostic v1

This package trains the already-defined Direct-C43 expert on all 799 source-2017 APKs and predicts the frozen 2018--2021 cohorts. It never loads future labels during prediction. Labels are joined only after `predictions_label_free.csv` has been written.

The experiment is diagnostic only. It cannot change the SVM router, Top25% budget, Direct-C43 architecture, threshold 0.5, or training settings. It writes only to `outputs/care-svm-direct-c43-future-v1`.
