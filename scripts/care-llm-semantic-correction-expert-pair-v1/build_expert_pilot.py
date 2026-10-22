#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from expert_pair_core import (
    EXPERT_SYSTEM_PROMPT, load_jsonl, prompt_hash, user_hash,
    validate_base_prompts,
)

BASE = Path("artifacts/care-llm-semantic-correction-prompt-v2/prompts_v2_pilot5_unlabeled.jsonl")
OUT = Path("artifacts/care-llm-semantic-correction-expert-pair-v1")


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    base = validate_base_prompts(load_jsonl(args.base))
    expert = []
    audit = []
    for row in base:
        new = dict(row)
        new["system_prompt"] = EXPERT_SYSTEM_PROMPT
        new["prompt_sha256"] = prompt_hash(EXPERT_SYSTEM_PROMPT, row["user_prompt"])
        expert.append(new)
        audit.append({
            "case_id": row["case_id"], "sha256": row["sha256"],
            "stage1_pred": row["stage1_pred"], "candidate_pred": row["candidate_pred"],
            "v2_prompt_sha256": row["prompt_sha256"],
            "v2p1_prompt_sha256": new["prompt_sha256"],
            "v2_user_prompt_sha256": user_hash(row["user_prompt"]),
            "v2p1_user_prompt_sha256": user_hash(new["user_prompt"]),
            "user_prompt_identical": row["user_prompt"] == new["user_prompt"],
            "only_system_prompt_changed": (
                row["case_id"] == new["case_id"] and row["sha256"] == new["sha256"]
                and row["stage1_pred"] == new["stage1_pred"]
                and row["candidate_pred"] == new["candidate_pred"]
                and row["user_prompt"] == new["user_prompt"]
                and row["system_prompt"] != new["system_prompt"]
            ),
            "ground_truth_loaded": 0, "fixed_broken_loaded": 0,
        })
    if not all(row["only_system_prompt_changed"] for row in audit):
        raise RuntimeError("BLOCKED_BY_CONTRACT: pair changed more than the system prompt")
    atomic_text(args.output / "prompts_v2p1_expert_pilot5_unlabeled.jsonl", "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in expert
    ))
    csv_path = args.output / "prompt_pair_audit.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
        writer.writeheader(); writer.writerows(audit)
    temporary.replace(csv_path)
    contract = {
        "status": "EXPERT_PAIR_BUILD_ONLY_NO_LLM_CALL", "pilot_rows": 5,
        "only_system_prompt_changed": True, "user_prompts_identical": True,
        "ground_truth_loaded": False, "fixed_broken_loaded": False,
        "future_rows_loaded": 0, "llm_calls_performed": 0,
    }
    atomic_text(args.output / "build_contract.json", json.dumps(contract, indent=2) + "\n")
    print("EXPERT_PAIR_BUILD_PASS")
    print("pilot_rows = 5")
    print("only_system_prompt_changed = true")
    print("ground_truth_loaded = false")
    print("llm_calls_performed = 0")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(str(error), flush=True); raise SystemExit(2)

