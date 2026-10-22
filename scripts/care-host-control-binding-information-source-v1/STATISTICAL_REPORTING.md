# Statistical reporting contract and reviewer-risk note

The reporting design was checked using the nature-statistics skill. This does
not constitute external statistical validation or preregistration of the
entire study. protocol.json freezes this version before its actual outcomes.

Independent reporting unit: APK (799 source, 200 routed final decisions).
Host/call-level observations are nested feature inputs, not independently
labelled training examples. Five folds are blocking/evaluation partitions, not
five independent datasets. Twenty null repeats are sensitivity realizations,
not 20 additional observations per APK.

Primary estimates: paired BA and the two Stage-I-side conditional AUC
differences, B2 versus aggregate/shuffle/placebo, plus BA versus Direct-C43.
Report all ten prespecified contrasts with ordinary paired 95% bootstrap
intervals and Bonferroni-adjusted intervals, not only favorable comparisons.
The 2000 resamples preserve outer fold, true label, route and Stage-I side.
Six fixed/broken/net difference intervals are exploratory and unadjusted.
All null performance realizations and each 95th-percentile envelope are
reported; the envelope is not an exact conditional-randomization p-value.

Missing/corrupt inputs block the whole test; no APK is excluded to improve
metrics. Undefined conditional AUCs remain unavailable, never zero or one.
Effect magnitude, damage and fold direction are required alongside intervals.
Both conditional sides must pass the frozen gate; one side is partial support.

Important limits: test predictions are strict outer OOF, but upstream training
hidden states are in-sample. Auxiliary parameters are consequently fixed with
no inner search or fully nested upstream refitting claim. Bootstrap inference
conditions on models and the cohort, without family-cluster or model-refit
uncertainty. Repeated historical inspection of source and prior negative
experiments means the source is development evidence, not pristine confirmation.
Within-version correction is not study-wide correction for all past tests.

No mechanistic or causal conclusion follows from a class association. Within-
APK synthetic swaps preserve specified marginals but are not executable APK
interventions. A local CFG proof is not consent, data purpose, runtime behavior
or positive benign evidence. A negative test only bounds the fixed compact
binding descriptor/readout; it does not prove static detection is impossible.

Software versions and measured runtimes are recorded from the actual server
environment, not assumed from local tests. Source paths, hashes, dimensions,
projection, seeds, exclusion policy and model parameters are exported.
