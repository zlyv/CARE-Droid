# CARE-Droid HostRel semantic-background diagnostics v1

Two explicitly separated development diagnostics compare locked H0 against
hard role-conditioned OriginalHost74 (B1) and HostRel74 (R1):

1. frozen temporal: fit on 2017 only, label-free inference on 2018-2021;
2. independent same-year five-fold OOF for each of 2018-2021.

Candidate validation Hosts and all three anomaly backgrounds use the same
74-D representation. No Ego tensor or Ego encoder is used. H0 predictions are
locked and reused. Both candidate arms keep Q-Query, DI25, max-delta 2.25,
32 shared prototypes, 24 benign prototypes, and 33,036 parameters.

The two GPUs run B1 and R1 concurrently; within-year jobs are checkpointed.
2018-2021 are already-inspected development years, not an independent final
temporal evaluation.

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-di25-hostrel-semantic-background-diagnostics-v1.zip -d scripts/
bash scripts/care-di25-hostrel-semantic-background-diagnostics-v1/launch_tmux.sh
tail -F artifacts/care-di25-hostrel-semantic-background-diagnostics-v1.console.log
```

Outputs:

- `artifacts/care-di25-hostrel-semantic-background-diagnostics-v1/frozen_results/REPORT.txt`
- `artifacts/care-di25-hostrel-semantic-background-diagnostics-v1/within_results/REPORT.txt`
- `artifacts/care-di25-hostrel-semantic-background-diagnostics-v1/REPORT.txt`
