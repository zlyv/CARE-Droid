#!/usr/bin/env python3
"""Audit feature-string labels against the frozen 799-row P72 labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from source_label_contract import source_label_of


def rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def norm(value) -> str:
    return str(value or "").strip().casefold()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = rows(args.folds)
    reference_ids = [norm(row.get("sha256")) for row in reference]
    feature_rows = [
        row for row in rows(args.features)
        if str(row.get("status", "")).strip() == "OK"
    ]
    by_sha: dict[str, dict] = {}
    duplicate_ids = []
    for row in feature_rows:
        sha = norm(row.get("sha256"))
        if not sha:
            continue
        if sha in by_sha:
            duplicate_ids.append(sha)
        by_sha[sha] = row

    unknown = []
    mismatches = []
    missing = []
    parsed = []
    for row in reference:
        sha = norm(row.get("sha256"))
        feature = by_sha.get(sha)
        if feature is None:
            missing.append(sha)
            continue
        try:
            actual = source_label_of(feature)
        except RuntimeError as exc:
            unknown.append({"sha256": sha, "error": str(exc)})
            continue
        expected = int(row["label"])
        parsed.append(actual)
        if actual != expected:
            mismatches.append({
                "sha256": sha, "feature_label": actual,
                "reference_label": expected,
                "metadata_label": feature.get("metadata_label", ""),
            })

    checks = {
        "reference_exact_unique_799": len(reference_ids) == 799 == len(set(reference_ids)),
        "feature_ids_unique": not duplicate_ids,
        "missing_reference_features_eq_0": not missing,
        "unknown_label_tokens_eq_0": not unknown,
        "label_mismatches_eq_0": not mismatches,
        "parsed_labels_exact_400_399": Counter(parsed) == Counter({0: 400, 1: 399}),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    audit = {
        "passed": all(checks.values()), "checks": checks,
        "accepted_tokens": {"benign": 0, "malicious": 1, "malware": 1},
        "missing": missing, "duplicate_ids": sorted(set(duplicate_ids)),
        "unknown": unknown, "mismatches": mismatches,
        "parsed_label_counts": dict(Counter(parsed)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not audit["passed"]:
        raise SystemExit("SOURCE_LABEL_CONTRACT_GATE_FAILED")
    print("SOURCE_LABEL_CONTRACT_GATE_PASS")


if __name__ == "__main__":
    main()
