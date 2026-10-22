# CARE-Droid resolvable-routing diagnostic v1

This is a read-only diagnostic over the existing strict 2017 OOF prediction
table.  It does not retrain or tune a router.

For each router and each frozen second layer (`BoundedCorrection`, `DirectL2`),
it reports:

- Error Recall at the observed budget;
- Repairable Error Recall at Budget (RER@B);
- Harmful Sample Capture;
- Unrepairable Error Capture;
- fixed, broken, net and correction precision;
- aggregate and per-fold results.

Run after reliability-calibrated router v2:

```bash
bash scripts/care-resolvable-router-diagnostic-v1/run_on_server.sh \
  2>&1 | tee artifacts/care-resolvable-router-diagnostic-v1.console.log
```

Package results:

```bash
zip -r care-resolvable-router-diagnostic-results-v1.zip \
  artifacts/care-resolvable-router-diagnostic-v1 \
  artifacts/care-resolvable-router-diagnostic-v1.console.log
```
