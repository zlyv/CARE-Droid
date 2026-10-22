# CARE-Droid Host-relative diagnostics v1

This package evaluates the locked H0 Q-Query representation against HM-R,
HC-R, and HMC-R under two complementary protocols:

1. frozen temporal diagnostic: fit on 2017 only, infer 2018-2021 without
   exposing future labels to training or inference;
2. independent same-year five-fold OOF: independently rebuild Stage-I,
   DI25, preprocessing, the three anomaly backgrounds, and Stage-II inside
   each year's outer folds.

For every candidate, the validation Host and all three anomaly backgrounds
(external benign, internal dominant, shared commonness) use the exact same
74-D candidate Host schema.  Background definitions and fold-safety rules do
not change.  H0 is reused from locked outputs and is never retrained.

No Ego tensor, upstream context, or downstream context is consumed.  The
router remains frozen DI25 and the correction remains Q-Query with
`max_delta=2.25`.

## Server run

The zip is uploaded to `/home/boyuanjie/projects/`, one level above the
project root:

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-di25-host-relative-diagnostics-v1.zip -d scripts/
bash scripts/care-di25-host-relative-diagnostics-v1/launch_tmux.sh
tail -F artifacts/care-di25-host-relative-diagnostics-v1.console.log
```

Two GPU workers are used.  Phase 1 runs four jobs concurrently (one frozen
and one within-year job on each GPU); phase 2 runs the two HMC jobs, one per
GPU.  Per-arm checkpoints make interrupted within-year jobs resumable.

## Outputs

- `frozen_results/REPORT.txt`
- `within_results/REPORT.txt`
- `REPORT.txt` and `COMBINED_DECISION.json`
- per-arm predictions, protocol audits, fit audits, and GPU logs

The combined rule is fixed in code before inspecting the new diagnostics.
Because 2018-2021 participate in method selection, they are development
evidence and cannot be called an unseen final evaluation.  Keep 2022-2024 as
the untouched final temporal holdout.
