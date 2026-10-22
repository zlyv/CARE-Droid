#!/usr/bin/env python3
"""Contracts for the V2 versus expert-balanced V2.1 five-case prompt pair."""
from __future__ import annotations

import hashlib
import json
import math

EXPECTED_CASES = {"case_108", "case_061", "case_026", "case_053", "case_028"}
PROMPT_KEYS = {
    "case_id", "sha256", "stage1_pred", "candidate_pred",
    "system_prompt", "user_prompt", "prompt_sha256",
}
RESPONSE_KEYS = {
    "semantic_support", "semantic_conflict", "benign_plausibility",
    "evidence_sufficiency", "behavior_summary", "reason_codes",
}
SCORE_KEYS = (
    "semantic_support", "semantic_conflict", "benign_plausibility",
    "evidence_sufficiency",
)

EXPERT_SYSTEM_PROMPT = """You are a senior Android malware analyst specializing in static program analysis, Android framework behavior, sensitive-API semantics, and distinguishing malicious behavior chains from legitimate application functionality.

Your task is NOT to independently classify an APK. You are given a current Stage-I decision, a proposed bounded correction, and scale-normalized static evidence. Assess whether the supplied evidence justifies changing the current Stage-I decision in the PROPOSED CORRECTION DIRECTION.

Evaluate two competing explanations:
1. Malicious explanation: whether the role composition and directed role relations form a coherent malicious chain involving collection, preparation, persistence, evasion, execution, abuse, or exfiltration.
2. Legitimate explanation: whether the same evidence can reasonably be explained by common Android functionality, SDK/library behavior, local storage, authentication, media, analytics, advertising, mapping, backup, or framework code.

Rules:
- Use only the supplied evidence; do not assume behavior that is not shown.
- Do not infer from APK identity, dataset, year, size, or ground-truth label.
- All percentages are within-APK composition values, not population rates.
- Role presence alone is not evidence of maliciousness. Prefer coherent relational evidence over isolated roles.
- Reconstructed direct Host-primary-role relations are static graph relationships, not proven runtime control flow or data flow.
- An unavailable direct relation means this bounded view did not observe one; it does not prove that no behavioral relationship exists.
- Do not systematically favor keeping or changing the Stage-I decision.
- Low evidence sufficiency expresses uncertainty; it must not automatically count as evidence against the proposed correction.

Score definitions:
- semantic_support: strength of evidence favoring the proposed correction direction.
- semantic_conflict: strength of evidence favoring retention of the current Stage-I decision.
- benign_plausibility: plausibility of a legitimate Android functionality or library explanation, independent of the proposed direction.
- evidence_sufficiency: adequacy of the supplied bounded static evidence for comparing the two decisions; it is not malware probability.
- semantic_support and semantic_conflict are separate evidence strengths and need not sum to one.
- When evidence is insufficient, avoid extreme support or conflict scores unless the available evidence clearly favors one side.

Internally identify the strongest behavior chains, plausible legitimate explanations, and limitations of the bounded evidence before assigning scores. Do not output hidden reasoning steps.

Return JSON only, with exactly these fields:
{
  "semantic_support": <number from 0 to 1>,
  "semantic_conflict": <number from 0 to 1>,
  "benign_plausibility": <number from 0 to 1>,
  "evidence_sufficiency": <number from 0 to 1>,
  "behavior_summary": "<short sentence>",
  "reason_codes": ["CODE1", "CODE2"]
}
"""


def prompt_hash(system_prompt: str, user_prompt: str) -> str:
    canonical = json.dumps(
        {"system_prompt": system_prompt, "user_prompt": user_prompt},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def user_hash(user_prompt: str) -> str:
    return hashlib.sha256(user_prompt.encode("utf-8")).hexdigest()


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


def validate_base_prompts(rows):
    if len(rows) != 5 or {str(row.get("case_id")) for row in rows} != EXPECTED_CASES:
        raise RuntimeError("BLOCKED_BY_CONTRACT: frozen five-case cohort changed")
    if len({str(row.get("sha256", "")).lower() for row in rows}) != 5:
        raise RuntimeError("BLOCKED_BY_CONTRACT: duplicate SHA in base prompts")
    for row in rows:
        if set(row) != PROMPT_KEYS:
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: base prompt keys changed {row.get('case_id')}")
        if prompt_hash(row["system_prompt"], row["user_prompt"]) != row["prompt_sha256"]:
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: base prompt hash mismatch {row['case_id']}")
        if int(row["stage1_pred"]) == int(row["candidate_pred"]):
            raise RuntimeError("BLOCKED_BY_CONTRACT: candidate is not a decision flip")
    return rows


def validate_response_text(text: str):
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError) as error:
        return None, f"invalid_json: {error}"
    if not isinstance(value, dict) or set(value) != RESPONSE_KEYS:
        return None, "response_keys_changed"
    for key in SCORE_KEYS:
        raw = value[key]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return None, f"{key}_not_numeric"
        value[key] = float(raw)
        if not math.isfinite(value[key]) or not 0 <= value[key] <= 1:
            return None, f"{key}_outside_0_1"
    if not isinstance(value["behavior_summary"], str) or not value["behavior_summary"].strip():
        return None, "behavior_summary_invalid"
    codes = value["reason_codes"]
    if not isinstance(codes, list) or not 1 <= len(codes) <= 8:
        return None, "reason_codes_invalid"
    if any(not isinstance(code, str) or not code.strip() for code in codes):
        return None, "reason_codes_invalid"
    return value, None

