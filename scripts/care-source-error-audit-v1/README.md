# CARE-Droid source error audit v1

This is a post-hoc descriptive audit. It does not retrain, select, or modify
Global24, A25, AsymA25, or the deduplicated bounded RGCondAttn expert.

It reports:

- false-negative/false-positive and confidence-stratified error rates;
- samples fixed, broken, or left wrong by each second-stage route;
- samples whose A25 and AsymA25 routing decisions differ;
- key-feature quartile error rates;
- standardized feature shifts between final errors and correct predictions.

Run from the repository root:

```bash
bash scripts/care-source-error-audit-v1/run_on_server.sh \
  2>&1 | tee artifacts/care-source-error-audit-v1.console.log
```

Feature associations are descriptive and must not be presented as causal.
