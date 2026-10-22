# CARE source local assets v2

This package performs only label-independent extraction for the fixed 799-APK
2017 source cohort. It writes atomic per-APK shards under `/mnt/HDD1`, supports
resume, owns an exclusive writer lock, and refuses to merge or unlock OOF
training unless the exact-lineage hard gate passes.

The cached Ego is a **sensitive-endpoint directed 1-hop Host Ego**: neighboring
nodes are distinct Host methods retained in the endpoint-centred bounded graph,
and edges are retained raw call edges followed by a Host-to-Host filter. It is
not a complete-call-graph Ego. The current model experiments read Host values
only; Ego assets are cached for a later predeclared ablation.

An APK that parses successfully but retains no sensitive endpoint is stored as
a valid zero-Host shard. It remains in the exact cohort and follows the
downstream deterministic missing-region path; it is not treated as an
extraction failure or removed from evaluation.
