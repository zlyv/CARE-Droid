#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import ssl
import urllib.error
import urllib.request
from pathlib import Path

from expert_pair_core import (
    EXPECTED_CASES, PROMPT_KEYS, load_jsonl, prompt_hash, validate_response_text,
)

INPUT = Path("artifacts/care-llm-semantic-correction-expert-pair-v1/prompts_v2p1_expert_pilot5_unlabeled.jsonl")
OUTPUT = Path("artifacts/care-llm-semantic-correction-expert-pair-v1")
URL = "https://api-inference.modelscope.cn/v1/chat/completions"
MODEL = "Qwen/Qwen3.8-Flash-Next"


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def validate_prompts(rows):
    if len(rows) != 5 or {row.get("case_id") for row in rows} != EXPECTED_CASES:
        raise RuntimeError("BLOCKED_BY_CONTRACT: expert pilot cohort changed")
    for row in rows:
        if set(row) != PROMPT_KEYS:
            raise RuntimeError("BLOCKED_BY_CONTRACT: expert prompt keys changed")
        if prompt_hash(row["system_prompt"], row["user_prompt"]) != row["prompt_sha256"]:
            raise RuntimeError("BLOCKED_BY_CONTRACT: expert prompt hash mismatch")


def request_once(url, key, payload, timeout):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST", headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "Accept": "application/json", "User-Agent": "care-droid-expert-pair-v1",
    })
    with urllib.request.urlopen(request, timeout=timeout, context=ssl.create_default_context()) as response:
        return int(response.status), response.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    rows = load_jsonl(args.input); validate_prompts(rows)
    if not args.execute:
        raise RuntimeError("BLOCKED_BY_CONTRACT: --execute required")
    key = os.environ.get("QWEN_API_KEY", "").strip()
    if not key:
        raise RuntimeError("BLOCKED_BY_CONTRACT: QWEN_API_KEY is unset")
    url = os.environ.get("QWEN_CHAT_COMPLETIONS_URL", URL).strip()
    model = os.environ.get("QWEN_MODEL", MODEL).strip()
    raw_path = args.output / "expert_responses_raw.jsonl"
    if raw_path.exists():
        raise RuntimeError("BLOCKED_BY_CONTRACT: response file already exists; refusing duplicate calls")
    results = []
    for index, row in enumerate(rows, 1):
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": row["system_prompt"]},
                {"role": "user", "content": row["user_prompt"]},
            ],
            "temperature": 0, "top_p": 1, "n": 1, "max_tokens": 500,
        }
        result = {
            "case_id": row["case_id"], "sha256": row["sha256"],
            "stage1_pred": int(row["stage1_pred"]),
            "candidate_pred": int(row["candidate_pred"]),
            "prompt_sha256": row["prompt_sha256"], "model": model,
            "attempt_number": 1,
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        try:
            status, body = request_once(url, key, payload, args.timeout)
            api = json.loads(body.decode("utf-8"))
            content = api["choices"][0]["message"]["content"]
            validated, error = validate_response_text(content)
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
        append_jsonl(raw_path, result); results.append(result)
        print(f"EXPERT_PAIR_ATTEMPT {index}/5 {row['case_id']} status={result['status']}", flush=True)

    flat = []
    for row in results:
        item = {key: row.get(key, "") for key in (
            "case_id", "sha256", "stage1_pred", "candidate_pred", "prompt_sha256", "status", "validation_error",
        )}
        parsed = row.get("validated_response") or {}
        for key in ("semantic_support", "semantic_conflict", "benign_plausibility", "evidence_sufficiency", "behavior_summary"):
            item[key] = parsed.get(key, "")
        item["reason_codes"] = "|".join(parsed.get("reason_codes", []))
        flat.append(item)
    csv_path = args.output / "expert_responses_validated.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader(); writer.writerows(flat)
    temporary.replace(csv_path)
    valid = sum(row["status"] == "VALID" for row in results)
    atomic_text(args.output / "expert_run_contract.json", json.dumps({
        "status": "EXPERT_PAIR_CALL_COMPLETE", "model": model,
        "network_attempts": 5, "valid_responses": valid, "automatic_retries": 0,
        "only_system_prompt_changed": True, "ground_truth_loaded": False,
        "fixed_broken_loaded": False, "care_predictions_modified": False,
    }, indent=2) + "\n")
    print("EXPERT_PAIR_FIVE_CALL_RUN_DONE")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(str(error), flush=True); raise SystemExit(2)

