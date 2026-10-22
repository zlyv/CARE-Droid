# CARE-Droid failure-type × R4-block audit v1

This is a read-only post-hoc diagnostic. It reuses already saved strict OOF
need-correction probe probabilities and locked historical M1 cohorts.

It does **not** retrain or replay CARE-Droid, fit new probes, search thresholds,
change DI25, or modify any prediction.

The four block probabilities are independently trained diagnostic probes from
the completed R4 component audit. They measure marginal OOF readability of a
target from each block. They must not be interpreted as additive attribution,
neural contribution, or causal suppression.

## Run

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o care-di25-hostego-failure-block-audit-v1.zip -d scripts/
bash scripts/care-di25-hostego-failure-block-audit-v1/launch_tmux.sh
tail -F artifacts/care-di25-hostego-failure-block-audit-v1.console.log
```

Main report:

```bash
cat artifacts/care-di25-hostego-dmd-bounded-v1/failure_block_audit/FAILURE_BLOCK_AUDIT_REPORT.txt
```
