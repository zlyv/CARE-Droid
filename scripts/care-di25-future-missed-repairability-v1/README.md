# CARE-Droid frozen DI25-missed repairability audit

Read-only counterfactual on the 2020/2021 DI25 misses using the already saved
Q_QUERY_ONLY bounded probabilities. No model is trained or rerun.

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-di25-future-missed-repairability-v1.zip -d scripts/
bash scripts/care-di25-future-missed-repairability-v1/run_on_server.sh \
  2>&1 | tee artifacts/care-di25-future-missed-repairability-v1.console.log
cat artifacts/care-di25-future-missed-repairability-v1/REPORT.txt
```
