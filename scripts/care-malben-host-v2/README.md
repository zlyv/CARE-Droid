# Host-only MalBen v2 source experiment

The only experimental change relative to the 33,388-parameter anomaly control
is the fold-local role-conditioned malicious-versus-benign evidence used by the
bounded correction. Global24, fixed outer folds, A25 Top-25%, Host-only input,
attention, FixKeep, and the logit-space bound `DeltaMax=2.25` remain fixed.

Each role/class has one outer-train centroid (`K=1`, `min_support=20`). Benign
and malware sides fall back independently to their own global class centroid.
Both class similarities for a region use one shared role scale; if either side
falls back, both use the outer-train global shared scale.
