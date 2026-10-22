# CARE-Droid future Stage-II failure mechanism audit

Read-only probability- and block-level diagnosis of the 46 jointly failed
2020/2021 samples. It uses saved Q/N0 outputs only.

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-di25-future-stage2-failure-mechanism-v1.zip -d scripts/
bash scripts/care-di25-future-stage2-failure-mechanism-v1/run_on_server.sh \
  2>&1 | tee artifacts/care-di25-future-stage2-failure-mechanism-v1.console.log
cat artifacts/care-di25-future-stage2-failure-mechanism-v1/REPORT.txt
```
