#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import ssl
import urllib.error
import urllib.request
from pathlib import Path

from preflight import MODEL, inspect
from stability_core import parse_response_text

OUTPUT = Path("artifacts/care-llm-semantic-correction-expert-stability-v1")
URL = "https://api-inference.modelscope.cn/v1/chat/completions"


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def request_once(url, key, payload, timeout):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST", headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "Accept": "application/json", "User-Agent": "care-droid-expert-stability-v1",
    })
    with urllib.request.urlopen(request, timeout=timeout, context=ssl.create_default_context()) as response:
        return int(response.status), response.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    prompts, _ = inspect()
    if not args.execute:
        raise RuntimeError("BLOCKED_BY_CONTRACT: --execute required")
    key = os.environ.get("QWEN_API_KEY", "").strip()
    if not key:
        raise RuntimeError("BLOCKED_BY_CONTRACT: QWEN_API_KEY is unset")
    url = os.environ.get("QWEN_CHAT_COMPLETIONS_URL", URL).strip()
    model = os.environ.get("QWEN_MODEL", MODEL).strip()
    if model != MODEL:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: model changed: {model}")
    raw_path = args.output / "repeat2_repeat3_raw.jsonl"
    if raw_path.exists():
        raise RuntimeError("BLOCKED_BY_CONTRACT: stability response file already exists")
    attempts = 0
    for repeat in (2, 3):
        for row in prompts:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": row["system_prompt"]},
                    {"role": "user", "content": row["user_prompt"]},
                ],
                "temperature": 0, "top_p": 1, "n": 1, "max_tokens": 500,
            }
            attempts += 1
            result = {
                "case_id": row["case_id"], "sha256": row["sha256"],
                "prompt_sha256": row["prompt_sha256"], "model": model,
                "repeat": repeat, "attempt_number": 1,
                "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            try:
                status, body = request_once(url, key, payload, args.timeout)
                api = json.loads(body.decode("utf-8"))
                content = api["choices"][0]["message"]["content"]
                validated, error = parse_response_text(content)
                result.update({
                    "http_status": status,
                    "raw_response_sha256": hashlib.sha256(body).hexdigest(),
                    "raw_api_response": api, "assistant_content": content,
                    "validated_response": validated, "validation_error": error,
                    "status": "VALID" if error is None else "INVALID_RESPONSE",
                })
            except urllib.error.HTTPError as error:
                body = error.read()
                result.update({
                    "status": "HTTP_ERROR", "http_status": int(error.code),
                    "error_body": body.decode("utf-8", errors="replace")[:4000],
                })
            except Exception as error:
                result.update({"status": "REQUEST_ERROR", "error_type": type(error).__name__, "error": str(error)})
            append_jsonl(raw_path, result)
            print(f"EXPERT_STABILITY_ATTEMPT {attempts}/10 repeat={repeat} {row['case_id']} status={result['status']}", flush=True)
    print("EXPERT_STABILITY_TEN_CALL_RUN_DONE")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(str(error), flush=True); raise SystemExit(2)

