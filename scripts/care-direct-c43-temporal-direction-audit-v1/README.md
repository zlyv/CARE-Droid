# Direct-C43 temporal conditional-direction audit

This read-only experiment determines whether frozen Direct-C43 failures from 2017–2021 are primarily caused by temporal loss of conditional ranking or by threshold/score overlap.

It reports FN-vs-TN and TP-vs-FP AUC with stratified bootstrap intervals for all and routed samples, expert-wrong confidence bins, and same-cohort oracle thresholds strictly as non-deployable diagnostic upper bounds.

It does not change the model, route, threshold, or prediction-side logic. Future data are diagnostic only.
