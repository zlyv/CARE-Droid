# CARE-Droid Qwen five-case semantic-correction pilot v1

This package sends the five manually inspected, unlabeled Prompt V2 records to
an OpenAI-compatible Qwen chat-completions endpoint. It performs one request per
case, has no automatic retry, never reads labels or FIXED/BROKEN outcomes, and
does not modify a CARE-Droid prediction.

Upload the zip into `/home/boyuanjie/projects/` and unpack it from the project:

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-llm-semantic-correction-qwen-pilot-v1.zip -d scripts/
```

Run the read-only checks first:

```bash
bash scripts/care-llm-semantic-correction-qwen-pilot-v1/run_preflight_only.sh
```

The supplied ModelScope page maps to model ID `Qwen/Qwen3.8-Flash-Next`.
The runner defaults to ModelScope's OpenAI-compatible chat endpoint:
`https://api-inference.modelscope.cn/v1/chat/completions`.

Enter only the ModelScope SDK Token, without writing it to a file:

```bash
read -rsp 'Qwen API key: ' QWEN_API_KEY; export QWEN_API_KEY; echo
```

The endpoint/model can still be overridden with `QWEN_CHAT_COMPLETIONS_URL`
and `QWEN_MODEL`, but do not change the model during this fixed pilot.

Run exactly five single-attempt calls:

```bash
bash scripts/care-llm-semantic-correction-qwen-pilot-v1/run_five_on_server.sh \
  2>&1 | tee artifacts/care-llm-semantic-correction-qwen-pilot-v1.console.log
```

Inspect:

```bash
cat artifacts/care-llm-semantic-correction-qwen-pilot-v1/REPORT.txt
column -s, -t artifacts/care-llm-semantic-correction-qwen-pilot-v1/responses_validated.csv
```

If a partial run exists, the runner refuses to add calls. Preserve the partial
output as audit evidence. Do not delete it and resume silently.
