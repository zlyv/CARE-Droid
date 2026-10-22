# CARE Role Reliability Gate v1

This source-only experiment audits and, only if predeclared diagnostic evidence
exists, trains a correction-acceptance gate on the locked P31-RF → RF-DI25 →
A0 Role C43-RQ line. It does not use API identity, HAIG, Host-call, or future
rows.

The expensive part is `nested_replay.py`: for every outer fold it regenerates
the four inner-fold C43 candidate predictions while excluding the outer test
fold. This is necessary because ordinary source OOF predictions from the other
folds were produced by models that saw the current outer-test fold and are not
valid gate-training data.

Run:

```bash
bash scripts/care-role-reliability-gate-v1/launch_tmux.sh
tail -F artifacts/care-role-reliability-gate-v1.console.log
```

If the diagnostic stopping rule fails, the pipeline intentionally skips neural
replay and reports `ROLE_RELIABILITY_GATE_NOT_SUPPORTED`.

