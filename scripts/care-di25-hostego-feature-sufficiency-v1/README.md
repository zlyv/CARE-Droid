# CARE-Droid Stage-II Feature Sufficiency Audit: checkpoint gate

This is a read-only prerequisite audit. It does not train CARE-Droid or a
diagnostic probe. It accepts only a complete 2018--2021 x five folds x three
seeds frozen scalar-head M1 checkpoint lineage with saved graph scaler and
anomaly fitting state. DMD B1/B2 checkpoints are explicitly rejected.

Run:

```bash
bash scripts/care-di25-hostego-feature-sufficiency-v1/run_checkpoint_gate.sh
```

If the gate reports `MISSING_FROZEN_M1_CHECKPOINTS`, the feature-sufficiency
experiment is not identifiable under the current no-retraining contract.
