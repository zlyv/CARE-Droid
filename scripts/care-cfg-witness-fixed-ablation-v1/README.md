# CARE-Droid CFG Witness Fixed-Capacity Ablation v1

Development-only 2017--2021 experiment. It reuses frozen APK-level CFG Witness
statistics and appends exactly three train-standardized values to Global24.
Stage-I, MeanOpposition Top25%, Host bags, RoleGraph, three backgrounds, BCE,
epochs, seeds and threshold remain unchanged.

Arms:

- W0_FROZEN_BASELINE: existing 24-dimensional Direct-C43 checkpoint.
- W1_WITNESS3: bidirectional-pair count, InterZ-uncovered Witness count and
  InterZ-uncovered fraction.
- W2_SIZE3_CONTROL: three size/call-count controls with the same 27-dimensional
  model capacity.
- W3_SIZE_CONDITIONAL_SHUFFLE: W1 values shuffled inside train-derived size
  strata, separately in train and validation, without labels.

No APK is opened. The future years are explicitly development data because
they were used to authorize these frozen Witness candidates.
