# CARE-Droid MalEval2026 frozen external evaluation v1

This package evaluates the fixed MalEval release (25 benign simulated false
positives, 200 archived MalRadar malware, 30 recent malware). It never selects
features, thresholds, routes, or models with MalEval labels.

The frozen r1 endpoint is the 2017+2018 calibrated P31-SVM and its
SVM-anchored MeanOpposition Top25% route. The target labels are joined only by
`evaluate.py`. SHA overlap with CARE 2017--2021 is audited and the SHA-disjoint
scope is primary.

Stage-I is checkpointed before the more expensive Stage-II builders run. Only
the 64 routed APKs receive Compact/C43 and endpoint-graph extraction. A routed
APK whose Stage-II assets fail retains its Stage-I probability; failures are
never silently deleted. `stage2_builders.json` records the exact frozen
builder/config paths.

`evaluate.py` runs once immediately after Stage-I, so a usable external SVM
result is retained even if a later routed Stage-II asset fails. It runs again
after C43. Single-class cohort BA/AUC are deliberately left undefined;
specificity/recall and Wilson intervals are the valid cohort measures.

