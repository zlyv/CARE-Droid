# CARE-Droid Role-conditioned API refinement v1

Strict source-only 2017 experiment. R0 reuses the locked A0 Role-only predictions. R1-R4 retain P31-RF, RF-anchored DI25, RoleGraph, Relation31, the three existing backgrounds and max_delta=2.25.

The API branch is built inside every outer-train fold: APK-frequency vocabulary, role-specific UNK columns and SVD16 are fit without validation labels or identities entering learned vocabulary selection. API evidence refines SelfRole31 before the unchanged Host encoder; it never enters prototype/background distances or changes localization relations.

Run:

```bash
bash scripts/care-role-conditioned-api-v1/launch_tmux.sh
tail -F artifacts/care-role-conditioned-api-v1.console.log
```

This package never opens 2018-2021 artifacts. A future/same-year diagnostic must be a separate frozen post-selection experiment.
