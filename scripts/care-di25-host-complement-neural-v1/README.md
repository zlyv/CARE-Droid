# CARE-Droid Host Complement Neural Integration v1

This isolated package evaluates three and only three additions to the frozen
M1 HostEgo/Anomaly/Bounded expert:

- `M_A_COMPOSITION`: unique-role unordered pair interactions within each Host.
- `M_B_POSITION`: fixed 13-D per-Host structural-position vector.
- `M_AB_COMPOSITION_POSITION`: the additive A+B residual, with two independent gates.

The locked M0 prediction file is never retrained. The original raw 74-D Host
rows alone enter the unchanged anomaly fitter. A/B are attached after the
original directed Host-Ego fusion and before Critical/Attention aggregation.
There is no C module, correction-head change, threshold search, hyperparameter
search, APK analysis, decompilation, or call-graph generation.

## Required staged workflow

```bash
cd /home/boyuanjie/projects/tasa-droid

/home/boyuanjie/envs/tif/bin/python \
  scripts/care-di25-host-complement-neural-v1/selftest.py

/home/boyuanjie/envs/tif/bin/python \
  scripts/care-di25-host-complement-neural-v1/preflight.py

/home/boyuanjie/envs/tif/bin/python \
  scripts/care-di25-host-complement-neural-v1/run_host_complement_experiment.py \
  --mode dry-run --arm all
```

Do not launch the full run unless all three commands print their PASS marker.
After human confirmation:

```bash
bash scripts/care-di25-host-complement-neural-v1/launch_tmux.sh
tail -F artifacts/care-di25-host-complement-neural-v1.console.log
```

The full runner is fold-resumable through prediction/diagnostic checkpoints.
It intentionally follows the inherited M1 convention of not persisting neural
state dictionaries; the checkpoint directories contain exact OOF probabilities,
diagnostics, fit audits, and fold metadata.

## Interpretation boundary

All results are 2018–2021 same-year five-fold OOF development/diagnostic
evidence. They are not future, unseen, zero-day, independent temporal, or final
holdout results. Mechanism cohorts are historical post-hoc diagnostics.

