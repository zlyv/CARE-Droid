# Conditional Host-path model sensitivity audit v4

This is an explicitly **artificial input perturbation**, not a full-graph Host deletion or program-causal intervention. It keeps each APK's frozen RoleGraph and Global24 context fixed, removes one local Host row, recomputes the dynamic Internal Background with the saved seed-specific prototypes and scalers, and runs the entire original Host encoder/attention/readout. A preselected 8-pair 2020 FN/TN pilot uses mean frozen attention Top-5, 20 within-APK random Host controls, same-primary-Role controls, and all three formal seeds. No training or threshold search.

Launch once from project root with `bash scripts/care-host-region-attribution-v5/launch_tmux.sh`. No output is overwritten.
