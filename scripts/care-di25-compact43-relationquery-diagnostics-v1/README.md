# CARE-Droid Compact43 Relation-Query diagnostics v1

Runs two frozen development diagnostics for the source-selected duplicate-free
`C43_RELATION_QUERY` candidate:

1. train on the fixed 799 APK 2017 source cohort and evaluate 2018--2021;
2. independent same-year five-fold OOF for each of 2018--2021.

GPU0 runs the frozen temporal job and GPU1 runs the same-year job concurrently.
The existing H0 predictions are locked controls and are never retrained.

`Compact43 = SelfRole31 + Tail12`. Directed `Relation31` affects only Host
attention and never directly enters background distance, Host value, summaries,
or the correction head. No Ego tensor is consumed.

The future years have already been inspected and these results are development
diagnostics, not unseen-future or independent final-test evidence.
