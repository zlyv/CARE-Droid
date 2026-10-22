#!/usr/bin/env python3
"""Contracts and metrics for the frozen expert-prompt stability audit."""
from __future__ import annotations

import hashlib
import json
import math
import re

EXPECTED_CASES = {"case_108", "case_061", "case_026", "case_053", "case_028"}
SCORE_KEYS = (
    "semantic_support", "semantic_conflict", "benign_plausibility",
    "evidence_sufficiency",
)
RESPONSE_KEYS = set(SCORE_KEYS) | {"behavior_summary", "reason_codes"}


def prompt_hash(system_prompt: str, user_prompt: str) -> str:
    canonical = json.dumps(
        {"system_prompt": system_prompt, "user_prompt": user_prompt},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_jsonl(path):
    if not path.is_file():
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: missing {path}")
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: invalid JSONL line {number}: {error}")
    return rows


def validate_response(value):
    if not isinstance(value, dict) or set(value) != RESPONSE_KEYS:
        raise RuntimeError("BLOCKED_BY_CONTRACT: validated response schema changed")
    result = dict(value)
    for key in SCORE_KEYS:
        raw = result[key]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: nonnumeric {key}")
        result[key] = float(raw)
        if not math.isfinite(result[key]) or not 0 <= result[key] <= 1:
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: invalid {key}")
    if not isinstance(result["behavior_summary"], str) or not result["behavior_summary"].strip():
        raise RuntimeError("BLOCKED_BY_CONTRACT: invalid behavior_summary")
    codes = result["reason_codes"]
    if not isinstance(codes, list) or not 1 <= len(codes) <= 8:
        raise RuntimeError("BLOCKED_BY_CONTRACT: invalid reason_codes")
    if any(not isinstance(code, str) or not code.strip() for code in codes):
        raise RuntimeError("BLOCKED_BY_CONTRACT: invalid reason code item")
    return result


def parse_response_text(text):
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError) as error:
        return None, f"invalid_json: {error}"
    try:
        return validate_response(value), None
    except RuntimeError as error:
        return None, str(error)


def accept_decision(value: dict) -> bool:
    """Predeclared conservative semantic-correction acceptance rule."""
    return bool(
        float(value["evidence_sufficiency"]) >= 0.5
        and float(value["semantic_support"]) >= 0.5
        and float(value["semantic_support"]) > float(value["semantic_conflict"])
    )


def normalize_code(code: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", code.upper()).strip("_")


def jaccard(left, right) -> float:
    a = {normalize_code(value) for value in left}
    b = {normalize_code(value) for value in right}
    return len(a & b) / len(a | b) if a | b else 1.0

