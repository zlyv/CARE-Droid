# CARE-Droid DMD-BC v1

One-shot, predeclared mechanism test that replaces only frozen M1's scalar
correction head. The 2018–2021 independent same-year five-fold OOF cohorts are
development/diagnostic sets because they were already inspected during method
diagnosis. They are not unseen future or temporal-generalization evidence.

Arms: B0 frozen M1 (reused), B1 DMD-BC-NoKeep, B2 DMD-BC-Full. Fixed seeds are
900000/900001/900002; max logit correction is 2.25. No sweep is implemented.

The runner first performs the 2018/fold-1/seed-900000 dry-run and emits
`DMD_BOUNDED_HEAD_DRYRUN_PASS`; only then does it enter the full experiment.
Every seed/fold writes an inference checkpoint and append-resumable internal
prediction rows. Long execution is launched in tmux.

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o care-di25-hostego-dmd-bounded-v1.zip -d scripts/
bash scripts/care-di25-hostego-dmd-bounded-v1/launch_tmux.sh
tail -F artifacts/care-di25-hostego-dmd-bounded-v1.console.log
```

Do not change the method after reading the diagnostic result. The final report
explicitly stops and requests human confirmation.

## Prediction-only KEEP-collapse diagnosis

After the one-shot experiment has completed, the following command performs
the authorized post-hoc analysis. It does not import a training runner, load a
model, create a checkpoint, tune a threshold, or alter existing predictions.

```bash
bash scripts/care-di25-hostego-dmd-bounded-v1/run_keep_collapse_posthoc.sh
```
