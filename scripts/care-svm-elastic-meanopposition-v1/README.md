# Elastic MeanOpposition Routing v1

This is a frozen, read-only routing audit for P31-SVM + Direct-C43.

- It derives one Q75 threshold from the label-free 2017 MeanOpposition score distribution.
- Source evaluation uses leave-one-fold-out thresholds.
- The deployment candidate is Q75 with a hard Top25 capacity cap, so it can contract but cannot expand beyond the existing budget.
- Future routes and the protocol hash are written before future labels are loaded.
- It compares against fixed Top25 and fixed equal-cost controls.
- It does not train a benefit gate, change Direct-C43, or select the descriptive Top15 result.
- Routed APK count is only a cost proxy; runtime is not inferred.
