# CARE-Droid TRCBE source-v1 label-free gate

This package performs only Stage A and the label-free routed-2017 coverage gate.

- It parses DEX class/method metadata but never builds a new call graph.
- It uses the frozen endpoint graph edges and exact frozen stable IDs.
- It does not use Manifest, fuzzy method matching, labels, future years, or train a classifier.
- The taxonomy is frozen before coverage and hashed.
- If coverage fails, sufficiency/null/end-to-end outputs are explicit `NOT_RUN` records.
- Formal files and source APKs are hashed before and after.

