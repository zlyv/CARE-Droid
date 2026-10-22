# CARE-Droid no-Ego decoupled correction-authority source audit v1

This is a strict-2017, read-only diagnostic that tests the proposed separation
of correction gate, direction and magnitude before authorizing another neural
training run.

- Gate: one of the existing strict OOF S0/S1/S2/S3 Need-Gate scores, threshold
  fixed at 0.5.
- Direction: opposite the frozen Stage-I hard decision. No label enters the
  forward rule.
- Magnitude: the minimum logit displacement required to cross the decision
  boundary by 0.05, capped by the already frozen 2.25 bound.
- DI25 route: canonical 200/799 source route.
- Ego: never read or instantiated.
- Training and threshold search: none.

Required inputs:

1. `artifacts/care-source-di25-reconstructed-v1/source_router_inputs.csv`
2. `artifacts/care-di25-noego-needgate-source-v1/oof_predictions.csv`

The second input must be created by the strict source-only Need-Gate package;
the preflight verifies its exact SHA set against the canonical DI25 route.

Upload the zip to `/home/boyuanjie/projects/`, then run:

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-di25-noego-decoupled-authority-source-v1.zip -d scripts/

bash scripts/care-di25-noego-decoupled-authority-source-v1/run_on_server.sh \
  2>&1 | tee artifacts/care-di25-noego-decoupled-authority-source-v1.console.log
```

This audit is short and intentionally does not use tmux. If the Need-Gate OOF
file is absent, it stops with `BLOCKED_BY_CONTRACT` rather than falling back to
any future/same-year implementation.
