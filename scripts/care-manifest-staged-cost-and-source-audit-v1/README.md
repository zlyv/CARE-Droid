# Candidate cost and 2017 error audit

No training, tuning, APK execution, cache mutation or checkpoint access.
Samples 21 actual U_REF10 acquisition candidates without using labels to choose APKs.
Source2017: one median-size acquired APK per fixed outer fold.
Future2018–2021: one deterministic SHA-hash selection per size quartile per year.
Runs unchanged Manifest L1 then DEX RefPresence L3 twice each; checks exact feature parity.
Runs existing compact/endpoint builders once each to new isolated directories.
Analyzes all changed 2017 routes, not just timed samples, using frozen predictions.
Output is component timing, not complete C43 end-to-end latency.

Set MSC_OUT to a fresh run directory. Run in tmux, then inspect final_report.md,
candidate_component_costs.csv, source_change_anatomy.csv and final_decision.json.
