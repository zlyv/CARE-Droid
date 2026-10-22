#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from qwen_pilot_core import EXPECTED_CASES, load_pilot, prompt_hash, validate_response_text


def main():
    rows = []
    for index, case in enumerate(sorted(EXPECTED_CASES)):
        system = "system"
        user = f"candidate {index}"
        rows.append({
            "case_id": case, "sha256": f"{index:064x}",
            "stage1_pred": index % 2, "candidate_pred": 1 - (index % 2),
            "system_prompt": system, "user_prompt": user,
            "prompt_sha256": prompt_hash(system, user),
        })
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "pilot.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        assert len(load_pilot(path)) == 5
    valid = {
        "semantic_support": 0.2, "semantic_conflict": 0.6,
        "benign_plausibility": 0.7, "evidence_sufficiency": 0.3,
        "behavior_summary": "Ambiguous static evidence.",
        "reason_codes": ["AMBIGUOUS_STATIC_EVIDENCE"],
    }
    parsed, error = validate_response_text(json.dumps(valid))
    assert error is None and parsed == valid
    assert validate_response_text("```json\n{}\n```")[1].startswith("invalid_json")
    invalid = dict(valid); invalid["semantic_support"] = 1.1
    assert validate_response_text(json.dumps(invalid))[1] == "semantic_support_outside_0_1"
    print("QWEN_PILOT_SELFTEST_PASS")


if __name__ == "__main__":
    main()

