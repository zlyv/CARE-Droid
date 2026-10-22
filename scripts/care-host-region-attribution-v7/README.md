# Conditional Host-path model sensitivity audit v7

This is an explicitly **artificial input perturbation**, not a full-graph Host deletion or program-causal intervention. It keeps each APK's frozen RoleGraph and Global24 context fixed, removes one local Host row, recomputes the dynamic Internal Background with the saved seed-specific prototypes and scalers, and runs the entire original Host encoder/attention/readout. It enumerates every Host row for the frozen 77 routed, Stage-I-predicted-benign 2020 APKs, with all three formal seeds. Each row is aligned to a cached method-level Host ID by exact Role31, endpoint-count and Relation31 checks. No training or threshold search. The 22 existing matched FN/TN pairs are descriptive controls, not a model-selection domain.

Launch once from project root with `bash scripts/care-host-region-attribution-v7/launch_tmux.sh`. No output is overwritten.
