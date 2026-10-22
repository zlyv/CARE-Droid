#!/usr/bin/env python3
"""Pure validation helpers for the five-call Qwen pilot."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

EXPECTED_CASES = {"case_108", "case_061", "case_026", "case_053", "case_028"}
REQUIRED_PROMPT_KEYS = {
    "case_id", "sha256", "stage1_pred", "candidate_pred",
    "system_prompt", "user_prompt", "prompt_sha256",
}
REQUIRED_RESPONSE_KEYS = {
    "semantic_support", "semantic_conflict", "benign_plausibility",
    "evidence_sufficiency", "behavior_summary", "reason_codes",
}
FORBIDDEN_TEXT = (
    "ground_truth", "ground truth", "fixed", "broken", "true label",
    "真实标签", "修复成功", "修复失败",
)
SCORE_KEYS = (
    "semantic_support", "semantic_conflict", "benign_plausibility",
    "evidence_sufficiency",
)


def prompt_hash(system_prompt: str, user_prompt: str) -> str:
    canonical = json.dumps(
        {"system_prompt": system_prompt, "user_prompt": user_prompt},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_pilot(path: Path) -> list[dict]:
    if not path.is_file():
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: missing pilot file: {path}")
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: invalid JSONL line {number}: {error}")
        if set(row) != REQUIRED_PROMPT_KEYS:
            raise RuntimeError(
                f"BLOCKED_BY_CONTRACT: prompt keys changed at line {number}: {sorted(row)}"
            )
        rows.append(row)
    if len(rows) != 5:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: pilot rows={len(rows)}, expected=5")
    if {str(row["case_id"]) for row in rows} != EXPECTED_CASES:
        raise RuntimeError("BLOCKED_BY_CONTRACT: pilot case set changed")
    if len({str(row["sha256"]).lower() for row in rows}) != 5:
        raise RuntimeError("BLOCKED_BY_CONTRACT: duplicate pilot SHA")
    if len({str(row["prompt_sha256"]).lower() for row in rows}) != 5:
        raise RuntimeError("BLOCKED_BY_CONTRACT: duplicate pilot prompt hash")
    for row in rows:
        if int(row["stage1_pred"]) == int(row["candidate_pred"]):
            raise RuntimeError("BLOCKED_BY_CONTRACT: candidate is not a decision flip")
        actual = prompt_hash(str(row["system_prompt"]), str(row["user_prompt"]))
        if actual != str(row["prompt_sha256"]).lower():
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: prompt hash mismatch {row['case_id']}")
        combined = (str(row["system_prompt"]) + "\n" + str(row["user_prompt"])).lower()
        found = [token for token in FORBIDDEN_TEXT if token in combined]
        if found:
            raise RuntimeError(
                f"BLOCKED_BY_CONTRACT: forbidden prompt text {row['case_id']}: {found}"
            )
    return rows


def validate_response_text(text: str) -> tuple[dict | None, str | None]:
    """Validate exact JSON. Deliberately does not strip Markdown fences."""
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError) as error:
        return None, f"invalid_json: {error}"
    if not isinstance(value, dict):
        return None, "response_is_not_object"
    if set(value) != REQUIRED_RESPONSE_KEYS:
        return None, f"response_keys_changed: {sorted(value)}"
    for key in SCORE_KEYS:
        raw = value[key]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return None, f"{key}_not_numeric"
        number = float(raw)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            return None, f"{key}_outside_0_1"
        value[key] = number
    if not isinstance(value["behavior_summary"], str) or not value["behavior_summary"].strip():
        return None, "behavior_summary_invalid"
    if len(value["behavior_summary"]) > 600:
        return None, "behavior_summary_too_long"
    codes = value["reason_codes"]
    if not isinstance(codes, list) or not 1 <= len(codes) <= 8:
        return None, "reason_codes_invalid_length"
    if any(not isinstance(code, str) or not code.strip() or len(code) > 80 for code in codes):
        return None, "reason_codes_invalid_item"
    return value, None

