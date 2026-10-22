# Frozen acquisition-routing cascade audit

No new fits, APK extraction, checkpoint changes, threshold searches or automatic follow-on runs.
Reads frozen preceding route masks and same-fold/source-only H0 expert predictions.
Primary U_REF10; all existing fourteen routes retained as descriptive controls.
Checks exact identities, labels, folds, ensemble arithmetic and old Top10/25 metric parity.
Evaluates source2017 and each frozen future year2018–2021, with fixed0.5 classification.
Reports repairs/damage in added/dropped routes and same-count uncertainty controls.

Timing reuses20 source APK measurements. Component extrapolation and fixed-alpha cost
sensitivity are explicitly hypothetical, not measured end-to-end latency.
No confidence-interval/significance claims; three seeds are not independent APK replicates.

Set MCA_OUT to a new run directory, run.sh --preflight-only, then run.sh in tmux.
Inspect final_report.md, seed_summary.csv, cost_sensitivity.csv, final_decision.json.
