#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SCORES = ("semantic_support", "semantic_conflict", "benign_plausibility", "evidence_sufficiency")
BASE = Path("artifacts/care-llm-semantic-correction-qwen-pilot-v1/responses_validated.csv")
EXPERT = Path("artifacts/care-llm-semantic-correction-expert-pair-v1/expert_responses_validated.csv")
OUT = Path("artifacts/care-llm-semantic-correction-expert-pair-v1")


def read_csv(path):
    if not path.is_file():
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: missing {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 5 or any(row.get("status") != "VALID" for row in rows):
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: expected five valid rows in {path}")
    return {row["case_id"]: row for row in rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=BASE)
    parser.add_argument("--expert", type=Path, default=EXPERT)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    old, new = read_csv(args.baseline), read_csv(args.expert)
    if set(old) != set(new):
        raise RuntimeError("BLOCKED_BY_CONTRACT: paired case sets differ")
    paired = []
    for case in sorted(old):
        if old[case]["sha256"] != new[case]["sha256"]:
            raise RuntimeError("BLOCKED_BY_CONTRACT: paired SHA mismatch")
        row = {"case_id": case, "sha256": old[case]["sha256"]}
        for score in SCORES:
            before, after = float(old[case][score]), float(new[case][score])
            row[f"v2_{score}"] = before
            row[f"v2p1_{score}"] = after
            row[f"delta_{score}"] = after - before
        row["v2_support_gt_conflict"] = float(old[case]["semantic_support"]) > float(old[case]["semantic_conflict"])
        row["v2p1_support_gt_conflict"] = float(new[case]["semantic_support"]) > float(new[case]["semantic_conflict"])
        row["ordering_flipped"] = row["v2_support_gt_conflict"] != row["v2p1_support_gt_conflict"]
        paired.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "V2_V2p1_PAIRED_COMPARISON.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader(); writer.writerows(paired)
    mean_delta = {score: sum(row[f"delta_{score}"] for row in paired) / 5 for score in SCORES}
    case028 = next(row for row in paired if row["case_id"] == "case_028")
    report = [
        "CARE-Droid V2 vs expert-balanced V2.1 paired prompt pilot", "",
        "Five identical APK/user-evidence prompts; only the system prompt changed.",
        "No labels or FIXED/BROKEN outcomes were loaded.", "",
        *[f"mean_delta_{score}: {mean_delta[score]:+.6f}" for score in SCORES],
        f"ordering_flip_count: {sum(row['ordering_flipped'] for row in paired)}/5",
        f"case_028_v2p1_evidence_sufficiency: {case028['v2p1_evidence_sufficiency']:.6f}",
        "", "Interpretation boundary: five calls assess prompt behavior, not detection performance or stability.",
    ]
    (args.output / "REPORT.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("EXPERT_PAIR_SUMMARY_PASS")
    print("\n".join(report))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(str(error), flush=True); raise SystemExit(2)

