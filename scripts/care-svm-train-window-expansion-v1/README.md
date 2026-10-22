# CARE-Droid SVM train-window expansion v1

Purpose: distinguish a small-sample/diversity explanation from a recency/distribution explanation for temporal C43 benign-side failure.

Arms:

- W17: existing frozen train-2017 reference.
- W18: train only 2018 (similar-size recency control).
- W1718: train pooled 2017+2018.

All arms use calibrated P31-SVM, SVM-anchored MeanOpposition Top25%, unchanged A0 Role C43-RQ, threshold 0.5 and max_delta 2.25. Test years are 2019, 2020 and 2021. Target labels are not used during fitting. Because all target years have already been observed in prior work, results are diagnostics only.

Only W1718 beating both W17 and W18 supports the claim that added sample diversity/size helps. W18 matching W1718 indicates recency/distribution dominates.
