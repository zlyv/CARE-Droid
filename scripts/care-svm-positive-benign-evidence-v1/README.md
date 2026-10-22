# CARE-Droid SVM Positive Benign Functional Certificate v1

Source-only gated experiment. It tests whether an observed non-sensitive code-function context can positively explain observed sensitive Roles under fold-safe benign statistics.

It explicitly forbids `low anomaly`, absence of malware evidence, and `1-malware_score` as benign evidence.

Pipeline:

1. reproduce the fixed 2017 SVM route (200 APKs; 43 routed FP; 41 routed FN);
2. build outer-fold positive-only context-to-Role lift;
3. evaluate FP-vs-TP against count-only and matched shuffle;
4. only when every evidence gate passes, evaluate a positive-evidence bounded benign correction;
5. authorize a separate frozen future package only if source evidence and integration both pass.

The package does not modify CARE-Droid assets and does not load future rows.
