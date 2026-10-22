# CARE-Droid P31-LR C43 role-direction probe (source v1)

This package performs the required **first-stage Direction Probe only**. It does not train or modify C43, does not load a future row, and does not alter P31-LR or the frozen routing-only committee DI25 route.

## Frozen protocol

- Source: 2017, 799 APKs, existing fixed five folds.
- Stage-I: calibrated P31 Logistic Regression only.
- Router: existing LR-anchored four-member MeanOpposition, foldwise Top25% (200 APKs).
- E0: frozen current C43 bounded-head logit correction.
- E1: outer-train-only, role-conditioned benign/malware standardized diagonal-shrinkage distance. `d_ben - d_mal > 0` always means malware.
- E2: E1 signed Host values aggregated with the current nonnegative background residual. Backgrounds cannot reverse an individual Host sign.
- E3: E1 signed Host values aggregated with a leave-one-host-out internal residual. Current and LOO internal distances use the same current-background scale, so their masking comparison is meaningful.
- The raw LOO distance is clamped to the raw current-distance lower bound before scaling. This enforces the exact leave-one-out monotonicity property while removing float32 centroid round-off only.
- Unsupported roles: ambiguous/excluded; no cross-role class center is substituted.

The frozen prediction artifact does not contain per-Host C43 attention weights. The probe therefore does not retrain/replay the neural attention or claim to implement `sum(alpha_i*s_i)`. It reports the explicit mean, max-absolute, current-reliability-weighted, and LOO-reliability-weighted aggregations. This keeps the requested mechanism test read-only and prevents a hidden model change.

## Run

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-p31lr-c43-role-direction-probe-source-v1.zip -d scripts/
bash scripts/care-p31lr-c43-role-direction-probe-source-v1/launch_tmux.sh
tail -F artifacts/care-p31lr-c43-role-direction-probe-source-v1.console.log
```

## Outputs

- `REPORT.txt`
- `direction_quality.csv`
- `per_fold_direction.csv`
- `failure_taxonomy.csv`
- `routed_apk_evidence.csv`
- `routed_host_evidence.csv`
- `role_reference_audit.csv`
- `internal_masking_audit.csv`
- `fold_audit.csv`
- `protocol.json`

`DIRECTION_PROBE_ADVANCE_CANDIDATE=NONE` is a stop result: do not add a new head or run a future diagnostic. A candidate advances only when FN accuracy does not decline, FP and overall accuracy improve over E0, and the improvement is not confined to fewer than three folds. If multiple candidates pass, the predeclared source-only tie break is overall accuracy, FP accuracy, FN accuracy, then lower fold dispersion.
