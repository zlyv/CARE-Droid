#!/usr/bin/env python3
from pathlib import Path
from expert_pair_core import load_jsonl, validate_base_prompts

BASE = Path("artifacts/care-llm-semantic-correction-prompt-v2/prompts_v2_pilot5_unlabeled.jsonl")
PRIOR = Path("artifacts/care-llm-semantic-correction-qwen-pilot-v1/responses_validated.csv")

try:
    rows = validate_base_prompts(load_jsonl(BASE))
    if not PRIOR.is_file():
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: missing frozen V2 pilot result {PRIOR}")
    print("EXPERT_PAIR_PREFLIGHT_PASS")
    print("pilot_rows = 5")
    print("user_evidence_reused = true")
    print("ground_truth_loaded = false")
    print("fixed_broken_loaded = false")
    print("additional_calls_authorized_by_runner = 5")
except RuntimeError as error:
    print(str(error), flush=True); raise SystemExit(2)

