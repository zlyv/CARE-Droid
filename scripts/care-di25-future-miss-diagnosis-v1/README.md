# CARE-Droid DI25 future miss diagnosis v1

Read-only post-hoc diagnosis of the frozen 2020/2021 DI25 misses. It does not
train or modify CARE-Droid and cannot be used to tune the already inspected
future diagnostic.

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-di25-future-miss-diagnosis-v1.zip -d scripts/
bash scripts/care-di25-future-miss-diagnosis-v1/run_on_server.sh \
  2>&1 | tee artifacts/care-di25-future-miss-diagnosis-v1.console.log
cat artifacts/care-di25-future-miss-diagnosis-v1/REPORT.txt
```

This is a short NumPy/CSV audit and does not require tmux or a GPU.
