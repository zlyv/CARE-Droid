#!/usr/bin/env python3
from __future__ import annotations

import csv
import itertools
import json
import statistics
from pathlib import Path

from preflight import FIRST, inspect
from stability_core import SCORE_KEYS, accept_decision, jaccard, load_jsonl, validate_response

ROOT = Path("artifacts/care-llm-semantic-correction-expert-stability-v1")
NEW = ROOT / "repeat2_repeat3_raw.jsonl"


def main():
    prompts, first = inspect()
    newer = load_jsonl(NEW)
    if len(newer) != 10 or any(row.get("status") != "VALID" for row in newer):
        raise RuntimeError("BLOCKED_BY_CONTRACT: expected ten valid new responses")
    by_case = {row["case_id"]: [] for row in first}
    for row in first:
        value = validate_response(row["validated_response"])
        by_case[row["case_id"]].append((1, value))
    for row in newer:
        value = validate_response(row["validated_response"])
        by_case[row["case_id"]].append((int(row["repeat"]), value))

    details, summary = [], []
    for case in sorted(by_case):
        runs = sorted(by_case[case])
        if [repeat for repeat, _ in runs] != [1, 2, 3]:
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: repeats incomplete for {case}")
        values = [value for _, value in runs]
        accepts = [accept_decision(value) for value in values]
        ordering = [value["semantic_support"] > value["semantic_conflict"] for value in values]
        pair_jaccard = [
            jaccard(values[left]["reason_codes"], values[right]["reason_codes"])
            for left, right in itertools.combinations(range(3), 2)
        ]
        row = {
            "case_id": case,
            "accept_repeat1": accepts[0], "accept_repeat2": accepts[1], "accept_repeat3": accepts[2],
            "accept_consistent": len(set(accepts)) == 1,
            "support_gt_conflict_consistent": len(set(ordering)) == 1,
            "reason_code_jaccard_mean": statistics.mean(pair_jaccard),
            "reason_code_jaccard_min": min(pair_jaccard),
        }
        for key in SCORE_KEYS:
            numbers = [value[key] for value in values]
            row[f"{key}_mean"] = statistics.mean(numbers)
            row[f"{key}_stdev"] = statistics.stdev(numbers)
            row[f"{key}_range"] = max(numbers) - min(numbers)
        summary.append(row)
        for repeat, value in runs:
            details.append({
                "case_id": case, "repeat": repeat, **{key: value[key] for key in SCORE_KEYS},
                "support_gt_conflict": value["semantic_support"] > value["semantic_conflict"],
                "accept_llm": accept_decision(value),
                "reason_codes": "|".join(value["reason_codes"]),
                "behavior_summary": value["behavior_summary"],
            })

    def write_csv(path, rows):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(ROOT / "stability_by_case.csv", summary)
    write_csv(ROOT / "all_15_responses.csv", details)
    max_range = max(row[f"{key}_range"] for row in summary for key in SCORE_KEYS)
    accept_consistent = all(row["accept_consistent"] for row in summary)
    ordering_consistent = all(row["support_gt_conflict_consistent"] for row in summary)
    case028 = next(row for row in summary if row["case_id"] == "case_028")
    case028_abstains = not any((case028[f"accept_repeat{i}"] for i in (1, 2, 3)))
    passed = max_range <= 0.15 and accept_consistent and ordering_consistent and case028_abstains
    report = [
        "CARE-Droid expert-balanced V2.1 three-repeat stability audit", "",
        "5 cases x 3 repeats; repeat 1 reused from the frozen paired pilot.",
        "New API calls: 10. Labels/FIXED/BROKEN loaded: NO.", "",
        f"maximum_score_range: {max_range:.6f} (criterion <= 0.15)",
        f"accept_decision_consistent_all_cases: {str(accept_consistent).lower()}",
        f"support_conflict_ordering_consistent_all_cases: {str(ordering_consistent).lower()}",
        f"case_028_abstained_all_three: {str(case028_abstains).lower()}",
        f"minimum_reason_code_jaccard: {min(row['reason_code_jaccard_min'] for row in summary):.6f} (descriptive only)",
        f"STABILITY_DECISION={'PASS' if passed else 'FAIL'}", "",
        "The predeclared Accept_LLM rule is sufficiency>=0.5 AND support>=0.5 AND support>conflict.",
        "This audit measures API/prompt repeatability, not correction accuracy.",
    ]
    (ROOT / "REPORT.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    (ROOT / "stability_contract.json").write_text(json.dumps({
        "status": "STABILITY_AUDIT_COMPLETE", "passed": passed,
        "cases": 5, "repeats_per_case": 3, "new_calls": 10,
        "score_range_limit": 0.15, "labels_loaded": False,
        "fixed_broken_loaded": False, "care_predictions_modified": False,
    }, indent=2) + "\n", encoding="utf-8")
    print("EXPERT_STABILITY_SUMMARY_DONE")
    print("\n".join(report))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(str(error), flush=True); raise SystemExit(2)

