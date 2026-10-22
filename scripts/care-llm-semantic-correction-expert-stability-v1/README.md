# CARE-Droid expert-balanced V2.1 stability audit v1

This audit reuses the five frozen repeat-1 expert responses and performs exactly
two additional calls per case (10 new calls). Prompt, model and decoding settings
are unchanged. It never reads labels or FIXED/BROKEN outcomes.

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-llm-semantic-correction-expert-stability-v1.zip -d scripts/
bash scripts/care-llm-semantic-correction-expert-stability-v1/run_preflight_only.sh
```

After PASS, enter the ModelScope token in memory and run:

```bash
read -rsp 'ModelScope key: ' QWEN_API_KEY; export QWEN_API_KEY; echo
bash scripts/care-llm-semantic-correction-expert-stability-v1/run_ten_on_server.sh \
  2>&1 | tee artifacts/care-llm-semantic-correction-expert-stability-v1.console.log
```

Inspect:

```bash
cat artifacts/care-llm-semantic-correction-expert-stability-v1/REPORT.txt
column -s, -t artifacts/care-llm-semantic-correction-expert-stability-v1/stability_by_case.csv
```
