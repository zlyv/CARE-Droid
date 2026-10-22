# CARE-Droid reliability-aware directional conflict router v1

This package changes only the Stage-I-to-Stage-II routing rule.  It keeps the
frozen Global24 probability, the deduplicated bounded local expert, the source
folds, the final threshold, and the 25% Stage-II budget unchanged.

## Main idea

The route distinguishes two reasons for sending an APK to Stage II:

1. **Boundary uncertainty（边界不确定性）**: Global24 is close to 0.5.
2. **Reliable evidence conflict（可靠证据冲突）**: at least one reliable
   semantic view confidently opposes the Global24 decision.

The four auxiliary views are Manifest, Code/Call Scale, Sensitive Behavior,
and Graph Statistics.  No fifth view is introduced.

For each view, source-only OOF predictions estimate:

- positive reliability: smoothed PPV, `P(y=1 | view predicts 1)`;
- negative reliability: smoothed NPV, `P(y=0 | view predicts 0)`.

Before computing contradiction strength, every auxiliary probability is Platt
calibrated using source-only OOF predictions.  In the strict source audit, each
outer fold fits calibration only from that fold's outer-training rows.

If Global24 predicts benign, only reliable malware-supporting view evidence is
counted.  If Global24 predicts malware, only reliable benign-supporting view
evidence is counted.  `SoftOR` then combines boundary risk and the maximum
directional conflict:

```text
U_route = 1 - (1 - U_boundary) * (1 - U_conflict)
```

This is a bounded monotone fusion score, not a calibrated probability.
View-sufficiency weights `q_iv` are deliberately fixed to one in v1.

## Compared routes

- `A25`: frozen boundary-only route;
- `ConflictMax25`: reliable directional conflict only;
- `SoftOR25`: primary boundary-plus-conflict route;
- `SoftOR_NoReliability25`: ablation with every reliability equal to one;
- `SoftOR_Top2Mean25`: aggregation ablation using the two strongest views.

The future diagnostic uses both annual equal-budget `YearTop25` routes and
2017-fitted `SourceQ75` thresholds.  Future labels are first accessed after all
scores, routes, and final probabilities have been frozen.

## Required existing server dependencies

The following established directories and artifacts must already exist:

- `scripts/care-multiview-conformal-router-source-v1`;
- `scripts/care-malcertain-router-future-v1`;
- the existing CARE source/future dependency scripts;
- `artifacts/p1-features`;
- `artifacts/p72-verified-dependency-source-ablation`;
- `artifacts/care-rgcondattn-dedup-bounded-source-v1`;
- `artifacts/care-rgcondattn-dedup-bounded-future-diagnostic-v1`.

## Run

From the project root:

```bash
bash scripts/care-directional-conflict-router-v1/run_on_server.sh \
  2>&1 | tee artifacts/care-directional-conflict-router-v1.console.log
```

The launcher prefers `/home/boyuanjie/envs/tif/bin/python`, then `python3`, and
finally `python`.  A different interpreter can be selected with `PYTHON_BIN`.

## Outputs

Source results are written to:

```text
artifacts/care-directional-conflict-router-source-v1/
```

Future diagnostic results are written to:

```text
artifacts/care-directional-conflict-router-future-v1/
```

Important files include routing and end-to-end metric tables, per-APK scores,
PPV/NPV reliability audits, A25 overlap counts, architecture/audit JSON files,
and compact `REPORT.txt` summaries.  Pay particular attention to
`high_confidence_error_recall`, because that measures the errors the new
conflict component is specifically intended to recover.

To package both result directories on the server:

```bash
zip -r care-directional-conflict-router-results.zip \
  artifacts/care-directional-conflict-router-source-v1 \
  artifacts/care-directional-conflict-router-future-v1 \
  artifacts/care-directional-conflict-router-v1.console.log
```
