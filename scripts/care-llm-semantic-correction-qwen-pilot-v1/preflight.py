#!/usr/bin/env python3
"""Read-only preflight for the fixed five-prompt pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen_pilot_core import load_pilot

DEFAULT_INPUT = Path(
    "artifacts/care-llm-semantic-correction-prompt-v2/"
    "prompts_v2_pilot5_unlabeled.jsonl"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    rows = load_pilot(args.input)
    summary = {
        "passed": True,
        "pilot_rows": len(rows),
        "unique_sha": len({row["sha256"] for row in rows}),
        "candidate_directions": sorted(
            f"{int(row['stage1_pred'])}->{int(row['candidate_pred'])}" for row in rows
        ),
        "ground_truth_loaded": False,
        "fixed_broken_loaded": False,
        "llm_calls_performed": 0,
        "maximum_calls_when_executed": 5,
        "automatic_retry": False,
    }
    print("QWEN_PILOT_PREFLIGHT_PASS")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(str(error), flush=True)
        raise SystemExit(2)

