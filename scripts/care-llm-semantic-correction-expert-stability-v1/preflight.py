#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from stability_core import EXPECTED_CASES, load_jsonl, prompt_hash, validate_response

PROMPTS = Path("artifacts/care-llm-semantic-correction-expert-pair-v1/prompts_v2p1_expert_pilot5_unlabeled.jsonl")
FIRST = Path("artifacts/care-llm-semantic-correction-expert-pair-v1/expert_responses_raw.jsonl")
MODEL = "Qwen/Qwen3.8-Flash-Next"


def inspect():
    prompts = load_jsonl(PROMPTS)
    first = load_jsonl(FIRST)
    if len(prompts) != 5 or {row.get("case_id") for row in prompts} != EXPECTED_CASES:
        raise RuntimeError("BLOCKED_BY_CONTRACT: expert prompt cohort changed")
    if len(first) != 5 or {row.get("case_id") for row in first} != EXPECTED_CASES:
        raise RuntimeError("BLOCKED_BY_CONTRACT: first-response cohort changed")
    prompt_by_case = {row["case_id"]: row for row in prompts}
    for row in prompts:
        if prompt_hash(row["system_prompt"], row["user_prompt"]) != row["prompt_sha256"]:
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: prompt hash mismatch {row['case_id']}")
    for row in first:
        case = row["case_id"]
        if row.get("status") != "VALID" or row.get("model") != MODEL:
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: invalid first response {case}")
        if row.get("prompt_sha256") != prompt_by_case[case]["prompt_sha256"]:
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: first-response prompt mismatch {case}")
        validate_response(row.get("validated_response"))
    return prompts, first


if __name__ == "__main__":
    try:
        prompts, first = inspect()
        print("EXPERT_STABILITY_PREFLIGHT_PASS")
        print(json.dumps({
            "pilot_cases": 5, "existing_valid_repeat1": 5,
            "new_repeats_per_case": 2, "authorized_new_calls": 10,
            "model": MODEL, "temperature": 0,
            "prompt_changed": False, "labels_loaded": False,
            "fixed_broken_loaded": False, "care_predictions_modified": False,
        }, indent=2))
    except RuntimeError as error:
        print(str(error), flush=True); raise SystemExit(2)

