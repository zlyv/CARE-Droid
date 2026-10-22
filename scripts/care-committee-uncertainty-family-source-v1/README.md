# CARE-Droid committee uncertainty family source audit v1

Read-only strict-2017 comparison using the already calibrated LR/RF/XGB/SVM
probabilities. It performs no model training and reads no future row.

From the repository root:

```bash
unzip -o ../care-committee-uncertainty-family-source-v1.zip -d scripts/
bash scripts/care-committee-uncertainty-family-source-v1/run_on_server.sh \
  2>&1 | tee artifacts/care-committee-uncertainty-family-source-v1.console.log
cat artifacts/care-committee-uncertainty-family-source-v1/REPORT.txt
```

This run is short and does not need tmux. Every router receives exactly 25% per
outer fold. SHA is only a deterministic tie breaker for coarse scores such as
vote entropy. Primary endpoint: Stage-I Error Recall@25.
