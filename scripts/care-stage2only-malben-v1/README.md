# Independent Stage-II-only MalBen

This is a strict 2017 outer-fold OOF experiment over all 799 APKs. The model
forward path is `forward(graph, regions)` and has no Global24, Stage-I
probability, router, bounded-correction, or FixKeep input. The neural auxiliary
head is trained only from Stage-II information. The reported final classifier
is fold-local `StandardScaler + L2 LogisticRegression` on
`[g_i, z_mal-z_ben, h_critical, e_mal-e_ben]`.

APKs without a real Host remain in the cohort through the established
deterministic one-row zero Host bag. Invocation rate is therefore 100% over the
same 799 APKs.
