# CARE-Droid SVM consensus selective correction v1

This package audits one symmetric, label-free-at-inference rule on the frozen 2017 source predictions:

- Stage-I: calibrated P31-SVM;
- router: SVM-anchored MeanOpposition DI25 Top25%;
- candidate: A0 Role C43-RQ bounded correction;
- selector: accept a class-changing C43 correction only when at least two of LR/RF/XGB agree with the proposed class.

The majority rule is primary and unanimity is a frozen control. A matched-random control preserves the accepted flip count within fold and proposed class. Future data are not loaded. This package is an audit only: it does not overwrite or retrain the formal pipeline.
