# CARE-Droid expert-balanced prompt paired pilot v1

This is a five-call paired comparison against the already completed V2 pilot.
The same five APKs, user evidence, model, temperature and decoding settings are
used. Only the system prompt changes. No labels or FIXED/BROKEN outcomes are
read, and no CARE-Droid prediction is changed.

Upload the zip into `/home/boyuanjie/projects/`, then:

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-llm-semantic-correction-expert-pair-v1.zip -d scripts/
bash scripts/care-llm-semantic-correction-expert-pair-v1/run_preflight_only.sh
```

After all checks pass, enter the existing ModelScope SDK Token in memory only:

```bash
read -rsp 'ModelScope key: ' QWEN_API_KEY; export QWEN_API_KEY; echo
bash scripts/care-llm-semantic-correction-expert-pair-v1/run_five_on_server.sh \
  2>&1 | tee artifacts/care-llm-semantic-correction-expert-pair-v1.console.log
```

Inspect:

```bash
cat artifacts/care-llm-semantic-correction-expert-pair-v1/REPORT.txt
column -s, -t artifacts/care-llm-semantic-correction-expert-pair-v1/V2_V2p1_PAIRED_COMPARISON.csv
```
