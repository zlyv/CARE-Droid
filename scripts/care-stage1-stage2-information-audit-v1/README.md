# CARE Stage-I robustness and Stage-II information-ceiling audit v1

This is a source-only diagnostic package. It does not alter or integrate a new CARE model.

It answers two questions:

1. How much of the 2017 P31-RF result survives low-capacity baselines, label-free P31 near-duplicate grouping, feature masking, and calibration audit?
2. Under identical low-capacity conditional probes, do frozen P31/Compact43/Role31/Relation31/full Host summaries contain bidirectional FN-vs-TN and FP-vs-TP information beyond count-only and label shuffle?

Important limits:

- The P31 grouping is explicitly a representation-near-duplicate audit, not a verified malware-family split.
- No future rows are loaded or used for selection.
- No probe output is connected to bounded correction.
- Existing formal inputs are hash-checked before and after the run.

Run with `bash scripts/care-stage1-stage2-information-audit-v1/launch_tmux.sh`.

