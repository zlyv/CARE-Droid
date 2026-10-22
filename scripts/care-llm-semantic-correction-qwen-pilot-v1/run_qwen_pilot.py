#!/usr/bin/env python3
"""Perform exactly one attempt for each of the five frozen pilot prompts."""
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

from qwen_pilot_core import load_pilot, validate_response_text

DEFAULT_INPUT = Path(
    "artifacts/care-llm-semantic-correction-prompt-v2/"
    "prompts_v2_pilot5_unlabeled.jsonl"
)
DEFAULT_OUTPUT = Path("artifacts/care-llm-semantic-correction-qwen-pilot-v1")
DEFAULT_MODELSCOPE_URL = "https://api-inference.modelscope.cn/v1/chat/completions"
DEFAULT_QWEN_MODEL = "Qwen/Qwen3.8-Flash-Next"


def atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_completed(path: Path) -> set[str]:
    completed = set()
    if not path.is_file():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            case = str(row["case_id"])
            if case in completed:
                raise RuntimeError(f"BLOCKED_BY_CONTRACT: duplicate saved response {case}")
            completed.add(case)
    return completed


def extract_content(response: dict) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("expected exactly one choice")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("missing choices[0].message.content")
    return message["content"]


def post_once(url: str, api_key: str, payload: dict, timeout: float) -> tuple[int, bytes]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "care-droid-qwen-pilot-v1",
        },
        method="POST",
    )
    # Default verified TLS context; no certificate bypass is allowed.
    with urllib.request.urlopen(request, timeout=timeout, context=ssl.create_default_context()) as response:
        return int(response.status), response.read()


def write_summary(output: Path, rows: list[dict], model: str, endpoint: str):
    csv_rows = []
    for row in rows:
        base = {
            "case_id": row["case_id"], "sha256": row["sha256"],
            "stage1_pred": row["stage1_pred"], "candidate_pred": row["candidate_pred"],
            "prompt_sha256": row["prompt_sha256"], "status": row["status"],
            "validation_error": row.get("validation_error", ""),
        }
        parsed = row.get("validated_response") or {}
        for key in (
            "semantic_support", "semantic_conflict", "benign_plausibility",
            "evidence_sufficiency", "behavior_summary",
        ):
            base[key] = parsed.get(key, "")
        base["reason_codes"] = "|".join(parsed.get("reason_codes", []))
        csv_rows.append(base)
    fields = list(csv_rows[0])
    csv_path = output / "responses_validated.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(csv_rows)
    temporary.replace(csv_path)
    valid = sum(row["status"] == "VALID" for row in rows)
    contract = {
        "status": "QWEN_PILOT_COMPLETE",
        "model": model,
        "endpoint_sha256": hashlib.sha256(endpoint.encode("utf-8")).hexdigest(),
        "pilot_rows": 5,
        "network_attempts": len(rows),
        "valid_responses": valid,
        "invalid_or_failed_responses": len(rows) - valid,
        "automatic_retries": 0,
        "ground_truth_loaded": False,
        "fixed_broken_loaded": False,
        "care_predictions_modified": False,
    }
    atomic_text(output / "pilot_contract.json", json.dumps(contract, indent=2) + "\n")
    report = [
        "CARE-Droid Qwen semantic-correction five-case pilot v1", "",
        f"model: {model}", "network attempts: 5", "automatic retries: 0",
        f"valid responses: {valid}/5", "ground-truth labels loaded: NO",
        "FIXED/BROKEN outcomes loaded: NO", "CARE-Droid predictions modified: NO", "",
        "This pilot evaluates response format and qualitative semantic behavior only.",
        "Five single calls do not establish run-to-run stability or detection performance.",
    ]
    atomic_text(output / "REPORT.txt", "\n".join(report) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    prompts = load_pilot(args.input)
    if not args.execute:
        raise RuntimeError("BLOCKED_BY_CONTRACT: --execute is required for network calls")
    endpoint = os.environ.get("QWEN_CHAT_COMPLETIONS_URL", DEFAULT_MODELSCOPE_URL).strip()
    api_key = os.environ.get("QWEN_API_KEY", "").strip()
    model = os.environ.get("QWEN_MODEL", DEFAULT_QWEN_MODEL).strip()
    if not endpoint.startswith("https://"):
        raise RuntimeError("BLOCKED_BY_CONTRACT: QWEN_CHAT_COMPLETIONS_URL must use https://")
    if not api_key:
        raise RuntimeError("BLOCKED_BY_CONTRACT: QWEN_API_KEY is unset")
    if not model:
        raise RuntimeError("BLOCKED_BY_CONTRACT: QWEN_MODEL is empty")

    raw_path = args.output / "responses_raw.jsonl"
    completed = read_completed(raw_path)
    if completed and completed != {str(row["case_id"]) for row in prompts}:
        raise RuntimeError(
            "BLOCKED_BY_CONTRACT: partial prior run exists; refusing to add calls. "
            "Preserve it as audit evidence and use a new output directory for a new authorized run."
        )
    if completed:
        print("QWEN_PILOT_ALREADY_COMPLETE")
        return

    request_manifest = []
    results = []
    for row in prompts:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": row["system_prompt"]},
                {"role": "user", "content": row["user_prompt"]},
            ],
            "temperature": 0,
            "top_p": 1,
            "n": 1,
            "max_tokens": 500,
        }
        request_bytes = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        request_manifest.append({
            "case_id": row["case_id"], "sha256": row["sha256"],
            "prompt_sha256": row["prompt_sha256"], "model": model,
            "request_sha256": sha256_bytes(request_bytes),
        })
        result = {
            "case_id": row["case_id"], "sha256": row["sha256"],
            "stage1_pred": int(row["stage1_pred"]),
            "candidate_pred": int(row["candidate_pred"]),
            "prompt_sha256": row["prompt_sha256"], "model": model,
            "attempt_number": 1,
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        try:
            status, body = post_once(endpoint, api_key, payload, args.timeout)
            result["http_status"] = status
            result["raw_response_sha256"] = sha256_bytes(body)
            result["raw_api_response"] = json.loads(body.decode("utf-8"))
            content = extract_content(result["raw_api_response"])
            result["assistant_content"] = content
            validated, error = validate_response_text(content)
            result["validated_response"] = validated
            result["validation_error"] = error
            result["status"] = "VALID" if error is None else "INVALID_RESPONSE"
        except urllib.error.HTTPError as error:
            body = error.read()
            result.update({
                "status": "HTTP_ERROR", "http_status": int(error.code),
                "raw_response_sha256": sha256_bytes(body),
                "error_body": body.decode("utf-8", errors="replace")[:4000],
            })
        except Exception as error:  # one attempt only; record and continue
            result.update({"status": "REQUEST_ERROR", "error_type": type(error).__name__, "error": str(error)})
        append_jsonl(raw_path, result)
        results.append(result)
        print(f"QWEN_PILOT_ATTEMPT {len(results)}/5 {row['case_id']} status={result['status']}", flush=True)

    atomic_text(
        args.output / "request_manifest.jsonl",
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in request_manifest),
    )
    write_summary(args.output, results, model, endpoint)
    print("QWEN_PILOT_FIVE_CALL_RUN_DONE")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(str(error), flush=True)
        raise SystemExit(2)
