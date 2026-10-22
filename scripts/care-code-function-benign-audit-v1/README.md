# CARE code-function benign evidence audit v1

Post-gate, source-only audit of the exploratory code-only FP-vs-TP AUC 0.693847.

It reuses the completed static cache and performs:

- removal of function families directly overlapping sensitive Roles;
- fold-level FP-vs-TP analysis;
- benign Role-frequency control;
- matched within-fold shuffle;
- size-correlation audit;
- stratified bootstrap confidence interval;
- leave-one-family-out analysis.

It does not rescan APKs, load future data, change the route, or integrate any correction.
