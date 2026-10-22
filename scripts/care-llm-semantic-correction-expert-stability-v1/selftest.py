#!/usr/bin/env python3
import json
from stability_core import accept_decision, jaccard, parse_response_text

base = {
    "semantic_support": .6, "semantic_conflict": .3,
    "benign_plausibility": .4, "evidence_sufficiency": .5,
    "behavior_summary": "Coherent relation.", "reason_codes": ["A", "B"],
}
parsed, error = parse_response_text(json.dumps(base))
assert error is None and accept_decision(parsed)
low = dict(base); low["evidence_sufficiency"] = .49
assert not accept_decision(low)
assert jaccard(["A", "B"], ["B", "C"]) == 1 / 3
print("EXPERT_STABILITY_SELFTEST_PASS")

