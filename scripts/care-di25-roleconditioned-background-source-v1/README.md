# CARE-Droid role-conditioned three-background source v1

Strict 2017 source-only, five-fold OOF experiment.  It freezes the original
H0 74-D Host representation, DI25, Q-Query, RoleToken, Global24/base-logit
correction context, bounded head, 32 shared prototypes, 24 benign prototypes,
and 33,036 neural parameters.

- B0: locked H0 generic background matching (not retrained).
- B1: hard dominant-role matched background selection.
- B2: soft multi-role compatibility-adjusted background selection.

The endpoint-role sidecar is constructed label-free from the frozen endpoint
cache.  It controls external-benign, internal-dominant, and shared-commonness
matching but never enters the neural correction vector. No Ego or future row
is loaded.

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-di25-roleconditioned-background-source-v1.zip -d scripts/
bash scripts/care-di25-roleconditioned-background-source-v1/launch_tmux.sh
tail -F artifacts/care-di25-roleconditioned-background-source-v1.console.log
```
