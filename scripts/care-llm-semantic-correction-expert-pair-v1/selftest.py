#!/usr/bin/env python3
import json
from expert_pair_core import EXPERT_SYSTEM_PROMPT, prompt_hash, validate_response_text

assert "senior Android malware analyst" in EXPERT_SYSTEM_PROMPT
assert "Do not systematically favor" in EXPERT_SYSTEM_PROMPT
assert "Low evidence sufficiency" in EXPERT_SYSTEM_PROMPT
assert prompt_hash("a", "b") == prompt_hash("a", "b")
valid = {
    "semantic_support": .4, "semantic_conflict": .3,
    "benign_plausibility": .7, "evidence_sufficiency": .2,
    "behavior_summary": "Ambiguous bounded evidence.",
    "reason_codes": ["AMBIGUOUS"],
}
assert validate_response_text(json.dumps(valid))[1] is None
assert validate_response_text("```json\n{}\n```")[1].startswith("invalid_json")
print("EXPERT_PAIR_SELFTEST_PASS")

