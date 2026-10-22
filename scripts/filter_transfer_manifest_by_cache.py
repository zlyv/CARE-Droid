#!/usr/bin/env python3
"""Keep only transfer-manifest rows present in an available A3 cache."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def read_cache_shas(path: Path) -> set[str]:
    result = set()
    with gzip.open(path.expanduser().resolve(strict=True), "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "OK" and row.get("sha256"):
                result.add(str(row["sha256"]).strip().lower())
    return result


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.expanduser().resolve(strict=True).open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"CSV has no header: {path}")
        fields = [str(value).lstrip("\ufeff") for value in reader.fieldnames]
        rows = [{str(key).lstrip("\ufeff"): value for key, value in row.items()} for row in reader]
    return fields, rows


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve(strict=True)
    cache_path = args.cache.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    fields, rows = read_csv(manifest_path)
    cache_shas = read_cache_shas(cache_path)
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("split_name", "")].append(row)

    kept = []
    summaries = []
    for split_name in sorted(groups):
        current = groups[split_name]
        original_train = [row for row in current if row.get("partition") == "train"]
        original_test = [row for row in current if row.get("partition") == "test"]
        train = [row for row in original_train if row.get("sha256", "").lower() in cache_shas]
        test = [row for row in original_test if row.get("sha256", "").lower() in cache_shas]
        kept.extend(train)
        kept.extend(test)
        labels = Counter(row.get("label", "") for row in test)
        summaries.append(
            {
                "split_name": split_name,
                "original_train_count": len(original_train),
                "cached_train_count": len(train),
                "original_test_count": len(original_test),
                "cached_test_count": len(test),
                "cached_test_class_counts": dict(sorted(labels.items())),
                "binary_metrics_eligible": len(labels) == 2,
                "has_train_and_test": bool(train and test),
            }
        )

    manifest_out = output / "partial_cache_split_manifest.csv"
    write_csv(manifest_out, fields, kept)
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "PARTIAL_A3_CACHE_TRANSFER_MANIFEST",
        "source_manifest": str(manifest_path),
        "cache": str(cache_path),
        "cache_ok_sha256_count": len(cache_shas),
        "source_manifest_row_count": len(rows),
        "retained_manifest_row_count": len(kept),
        "split_summaries": summaries,
        "outputs": {"split_manifest": str(manifest_out)},
        "blocking_reasons": [
            "partial_cache_is_not_a_representative_final_test_set",
            "near_duplicate_and_package_lineage_cross_source_audit_not_applied",
            "partial_results_must_not_be_compared_as_final_accuracy",
        ],
        "notes": [
            "Rows absent from the current cache are removed from both train and test partitions.",
            "A split is binary-metric eligible only when both test labels are present.",
            "Use this output for smoke/early signal checks only.",
        ],
    }
    (output / "partial_cache_manifest_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
