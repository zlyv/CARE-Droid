# Frozen same-year error diagnosis

This is a read-only diagnostic for the already completed independent same-year
five-fold OOF predictions (2018--2021). It does not train a model, change the
DI25 route, tune the 25% budget, alter `max_delta_logit=2.25`, or overwrite an
existing prediction.

Run after unzipping the package into `scripts/`:

```bash
cd /home/boyuanjie/projects/tasa-droid
bash scripts/care-di25-hostego-v1/launch_error_diagnosis_tmux.sh
tail -F artifacts/care-di25-hostego-v1/error_diagnosis.console.log
```

The analysis fails closed unless the exact 1989-SHA router/M0/M1 cohorts,
year counts, Host/Ego cache gate, and known pooled BA/FP/FN values agree.

Important evidence boundary:

> These are independent same-year OOF diagnostics and must not be described as
> temporal-generalization, future-year, zero-day, or unseen-future performance.

The completed same-year run did not persist per-APK raw delta, gamma,
attention, critical-host, or prototype-background tensors. The diagnostic
therefore marks those fields unavailable instead of retraining or fabricating
them. The effective bounded logit change is recovered exactly from the saved
Stage-I and bounded-expert probabilities.
