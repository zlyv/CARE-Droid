#!/usr/bin/env python3
"""Create and hard-audit the missing future Compact76 feature cache.

This is intentionally separate from the already completed Host/Ego extraction.
It makes one resumable compact-feature pass over the frozen 2018--2021 lineage,
stores the result on the large HDD, and refuses to continue on duplicate,
outside-lineage, failed, or missing rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


EXPECTED_YEARS = {2018: 793, 2019: 399, 2020: 397, 2021: 400}


def norm(value: object) -> str:
    return str(value or "").strip().lower()


def row_id(row: dict[str, str]) -> str:
    return norm(row.get("sha256") or row.get("sample_id") or row.get("apk_id"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duplicates(ids: list[str]) -> dict[str, int]:
    return {key: count for key, count in Counter(ids).items() if key and count > 1}


def inspect(lineage: Path, output: Path) -> tuple[dict, bool]:
    lineage_rows = read_csv(lineage)
    lineage_ids = [row_id(row) for row in lineage_rows]
    years = [int(row.get("year") or row.get("metadata_year") or 0) for row in lineage_rows]
    year_counts = dict(Counter(years))

    feature_path = output / "features.csv"
    failure_path = output / "feature_failures.csv"
    feature_rows = read_csv(feature_path)
    failure_rows = read_csv(failure_path)
    feature_ids = [row_id(row) for row in feature_rows]
    failure_ids = [row_id(row) for row in failure_rows]
    wanted = set(lineage_ids)
    actual = set(feature_ids)

    audit = {
        "lineage": str(lineage.resolve()),
        "output_root": str(output.resolve()),
        "expected": 1989,
        "lineage_rows": len(lineage_rows),
        "lineage_unique_ids": len(set(lineage_ids)),
        "year_counts": {str(key): value for key, value in sorted(year_counts.items())},
        "feature_rows": len(feature_rows),
        "feature_unique_ids": len(actual),
        "failure_rows": len(failure_rows),
        "duplicate_feature_ids": duplicates(feature_ids),
        "duplicate_failure_ids": duplicates(failure_ids),
        "missing_ids": sorted(wanted - actual),
        "outside_lineage_ids": sorted(actual - wanted),
        "failed_ids": sorted(set(failure_ids)),
        "non_ok_feature_ids": sorted(
            row_id(row) for row in feature_rows
            if str(row.get("status", "")).strip() != "OK"
        ),
    }
    audit["checks"] = {
        "lineage_exact_unique_1989": len(lineage_rows) == len(set(lineage_ids)) == 1989,
        "future_year_counts_exact": year_counts == EXPECTED_YEARS,
        "feature_exact_unique_1989": len(feature_rows) == len(actual) == 1989,
        "duplicate_feature_ids_eq_0": not audit["duplicate_feature_ids"],
        "duplicate_failure_ids_eq_0": not audit["duplicate_failure_ids"],
        "missing_ids_eq_0": not audit["missing_ids"],
        "outside_lineage_ids_eq_0": not audit["outside_lineage_ids"],
        "failure_rows_eq_0": not failure_rows,
        "all_feature_status_ok": not audit["non_ok_feature_ids"],
    }
    audit["passed"] = all(audit["checks"].values())
    output.mkdir(parents=True, exist_ok=True)
    (output / "compact_feature_gate.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["CARE-Droid future Compact76 hard gate", "", f"passed: {str(audit['passed']).lower()}"]
    lines.extend(f"{name}: {str(value).lower()}" for name, value in audit["checks"].items())
    lines.extend([
        f"feature_rows: {len(feature_rows)}",
        f"failure_rows: {len(failure_rows)}",
        f"missing_ids: {len(audit['missing_ids'])}",
        f"outside_lineage_ids: {len(audit['outside_lineage_ids'])}",
    ])
    (output / "compact_feature_gate.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit, bool(audit["passed"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extractor", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--priority", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    for path in (args.lineage, args.extractor, args.mapping, args.priority):
        if not path.is_file():
            raise RuntimeError("missing future Compact76 dependency: " + str(path))

    pre, passed = inspect(args.lineage, args.output)
    if passed:
        print("FUTURE_COMPACT76_GATE_PASS (reusing complete HDD cache)")
        return

    # Resume is safe only for a clean subset. Duplicate/outside/failure rows
    # indicate a contaminated cache and must never be hidden by deduplication.
    fatal = (
        pre["duplicate_feature_ids"] or pre["duplicate_failure_ids"]
        or pre["outside_lineage_ids"] or pre["failed_ids"]
        or not pre["checks"]["lineage_exact_unique_1989"]
        or not pre["checks"]["future_year_counts_exact"]
    )
    if fatal:
        raise RuntimeError(
            "Future Compact76 cache is contaminated or the lineage changed. "
            "See compact_feature_gate.json. Use a new CARE_FUTURE_COMPACT_ROOT; "
            "rows were not silently deduplicated or deleted."
        )

    print(
        "Future Compact76 cache is incomplete: "
        f"completed={pre['feature_unique_ids']} pending={len(pre['missing_ids'])}. "
        "Starting/resuming one-time lightweight extraction on the HDD.",
        flush=True,
    )
    command = [
        sys.executable, str(args.extractor),
        "--lineage", str(args.lineage),
        "--mapping", str(args.mapping),
        "--priority", str(args.priority),
        "--output", str(args.output),
        "--progress-every", str(max(1, args.progress_every)),
        "--min-free-gb", "5",
    ]
    subprocess.run(command, check=True)
    post, passed = inspect(args.lineage, args.output)
    if not passed:
        raise RuntimeError(
            "Future Compact76 extraction finished but the hard gate failed. "
            f"features={post['feature_rows']} failures={post['failure_rows']} "
            f"missing={len(post['missing_ids'])}."
        )
    schema = args.output / "feature_schema.json"
    if not schema.is_file():
        raise RuntimeError("future feature_schema.json was not generated")
    post_manifest = {
        "features": str((args.output / "features.csv").resolve()),
        "features_sha256": sha256(args.output / "features.csv"),
        "schema": str(schema.resolve()),
        "schema_sha256": sha256(schema),
        "mapping_sha256": sha256(args.mapping),
        "priority_sha256": sha256(args.priority),
        "reused_host_ego_assets": True,
        "host_ego_reextracted": False,
    }
    (args.output / "compact_feature_manifest.json").write_text(
        json.dumps(post_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FUTURE_COMPACT76_GATE_PASS")


if __name__ == "__main__":
    main()
