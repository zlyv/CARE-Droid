# SVM+C43 removal of bounded correction v1

Strict 2017 source-only comparison:

- `B0_FORMAL_BOUNDED`: frozen `2.25*tanh(raw_delta)` correction.
- `U1_UNBOUNDED_RESIDUAL`: `base_logit + raw_delta`, with no tanh or clipping.
- `U2_DIRECT_STAGE2`: direct C43 logit on routed APKs; Stage-I logit is removed from the Stage-II input and FixKeep is disabled.

Stage-I SVM, fixed folds, MeanOpposition Top25% route, C43 inputs, seeds and Host localization remain matched. Future years are not loaded in this package.
