#!/usr/bin/env python3
"""Build an old-2017-to-new-unpacked transfer-evaluation manifest.

The output is deliberately limited to the requested transfer experiment:
legacy source rows from one training year and new-source rows from selected
test years.  Rows with an exact SHA256 collision between the training and test
partitions are excluded from the test partition and reported in the audit.

This script does not fit a model and does not inspect test labels for feature
selection.  The labels are copied only into the split manifest for later
evaluation and post-hoc metric computation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


LABELS = {"benign", "malicious", "malware"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.expanduser().resolve(strict=True).open(
        "r", encoding="utf-8-sig", errors="replace", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"CSV has no header: {path}")
        fields = [str(value).lstrip("\ufeff") for value in reader.fieldnames]
        rows = []
        for raw in reader:
            rows.append({str(key).lstrip("\ufeff"): value for key, value in raw.items()})
    return fields, rows


def norm_sha(row: dict[str, str]) -> str:
    value = str(row.get("sha256", "")).strip().lower()
    if len(value) != 64:
        raise SystemExit(f"invalid SHA256 in feature row: {value!r}")
    return value


def norm_label(row: dict[str, str]) -> str:
    value = str(row.get("metadata_label", "")).strip().lower()
    if value == "malware":
        return "malicious"
    if value not in {"benign", "malicious"}:
        raise SystemExit(f"unsupported metadata_label: {value!r}")
    return value


def norm_year(row: dict[str, str]) -> int:
    try:
        return int(str(row.get("metadata_year", "")).strip())
    except ValueError as exc:
        raise SystemExit(f"invalid metadata_year: {row.get('metadata_year')!r}") from exc


def eligible(row: dict[str, str]) -> bool:
    status = str(row.get("status", "OK")).strip().upper()
    return status == "OK" and str(row.get("metadata_label", "")).strip().lower() in LABELS


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-features", required=True, type=Path)
    parser.add_argument("--new-features", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--old-source-id", default="legacy_packed_small")
    parser.add_argument("--new-source-id", default="new_nonpacked_2017_2021")
    parser.add_argument("--train-year", type=int, default=2017)
    parser.add_argument("--test-years", nargs="+", type=int, default=[2018, 2019, 2020])
    parser.add_argument(
        "--include-new-train-year",
        action="store_true",
        help="also add new-source rows from --train-year to every training partition",
    )
    args = parser.parse_args()

    old_path = args.old_features.expanduser().resolve(strict=True)
    new_path = args.new_features.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    old_fields, old_rows = read_csv(old_path)
    new_fields, new_rows = read_csv(new_path)
    if old_fields != new_fields:
        old_only = sorted(set(old_fields) - set(new_fields))
        new_only = sorted(set(new_fields) - set(old_fields))
        raise SystemExit(
            "old/new feature schemas differ; rerun with the same extractor schema.\n"
            f"old_only={old_only}\nnew_only={new_only}"
        )

    for name, rows in (("old", old_rows), ("new", new_rows)):
        seen: set[str] = set()
        for row in rows:
            sha = norm_sha(row)
            if sha in seen:
                raise SystemExit(f"duplicate SHA256 inside {name} feature CSV: {sha}")
            seen.add(sha)

    old_train = []
    for row in old_rows:
        if (
            eligible(row)
            and str(row.get("source_id", "")).strip() == args.old_source_id
            and norm_year(row) == args.train_year
        ):
            old_train.append(row)

    training_rows = list(old_train)
    training_shas = {norm_sha(row) for row in training_rows}
    duplicate_training_exclusions = []
    new_train_candidates = []
    if args.include_new_train_year:
        if args.train_year in args.test_years:
            raise SystemExit("--train-year cannot also be listed in --test-years when new training is enabled")
        for row in new_rows:
            if (
                eligible(row)
                and str(row.get("source_id", "")).strip() == args.new_source_id
                and norm_year(row) == args.train_year
            ):
                new_train_candidates.append(row)
                sha = norm_sha(row)
                if sha in training_shas:
                    duplicate_training_exclusions.append(
                        {
                            "sha256": sha,
                            "year": args.train_year,
                            "label": norm_label(row),
                            "reason": "exact_sha256_duplicate_with_old_training_row",
                        }
                    )
                    continue
                training_rows.append(row)
                training_shas.add(sha)

    transfer_rows = list(training_rows)
    manifest_rows: list[dict[str, object]] = []
    split_summaries = []
    excluded_exact = list(duplicate_training_exclusions)

    new_by_year: dict[int, list[dict[str, str]]] = {year: [] for year in args.test_years}
    for row in new_rows:
        if (
            eligible(row)
            and str(row.get("source_id", "")).strip() == args.new_source_id
            and norm_year(row) in new_by_year
        ):
            new_by_year[norm_year(row)].append(row)

    for year in args.test_years:
        candidates = new_by_year[year]
        test_rows = []
        for row in candidates:
            sha = norm_sha(row)
            if sha in training_shas:
                excluded_exact.append(
                    {
                        "sha256": sha,
                        "year": year,
                        "label": norm_label(row),
                        "reason": "exact_sha256_in_old_training_partition",
                    }
                )
                continue
            test_rows.append(row)
            transfer_rows.append(row)

        split_name = f"transfer_old_{args.train_year}__new_{year}"
        for row in training_rows:
            sha = norm_sha(row)
            source_id = str(row.get("source_id", "")).strip()
            manifest_rows.append(
                {
                    "split_name": split_name,
                    "partition": "train",
                    "sha256": sha,
                    "year": args.train_year,
                    "label": norm_label(row),
                    "source_id": source_id,
                    "blocking_group": f"sha256:{sha}",
                    "split_reason": (
                        "old_source_fixed_training_year"
                        if source_id == args.old_source_id
                        else "new_source_same_year_added_to_training"
                    ),
                }
            )
        for row in test_rows:
            sha = norm_sha(row)
            manifest_rows.append(
                {
                    "split_name": split_name,
                    "partition": "test",
                    "sha256": sha,
                    "year": year,
                    "label": norm_label(row),
                    "source_id": args.new_source_id,
                    "blocking_group": f"sha256:{sha}",
                    "split_reason": "new_unpacked_cross_source_test",
                }
            )

        labels = Counter(norm_label(row) for row in test_rows)
        split_summaries.append(
            {
                "split_name": split_name,
                "test_year": year,
                "test_count": len(test_rows),
                "test_class_counts": dict(sorted(labels.items())),
                "candidate_count_before_exact_exclusion": len(candidates),
                "exact_duplicate_excluded_count": len(candidates) - len(test_rows),
                "binary_metrics_eligible": len(labels) == 2,
            }
        )

    # A single merged feature CSV is enough for the existing compact evaluator.
    merged_path = output / "transfer_features.csv"
    write_csv(merged_path, old_fields, transfer_rows)
    manifest_path = output / "transfer_split_manifest.csv"
    manifest_fields = [
        "split_name", "partition", "sha256", "year", "label", "source_id",
        "blocking_group", "split_reason",
    ]
    write_csv(manifest_path, manifest_fields, manifest_rows)
    excluded_path = output / "exact_duplicate_exclusions.csv"
    write_csv(excluded_path, ["sha256", "year", "label", "reason"], excluded_exact)

    old_labels = Counter(norm_label(row) for row in old_train)
    new_train_rows = [
        row for row in training_rows
        if str(row.get("source_id", "")).strip() == args.new_source_id
    ]
    training_labels = Counter(norm_label(row) for row in training_rows)
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "CROSS_SOURCE_TRANSFER_SPLIT_OLD_TRAIN_TO_NEW_TEST",
        "old_features_csv": str(old_path),
        "old_features_sha256": sha256_file(old_path),
        "new_features_csv": str(new_path),
        "new_features_sha256": sha256_file(new_path),
        "old_source_id": args.old_source_id,
        "new_source_id": args.new_source_id,
        "fixed_train_year": args.train_year,
        "include_new_train_year": bool(args.include_new_train_year),
        "training_source_ids": sorted({str(row.get("source_id", "")).strip() for row in training_rows}),
        "test_years": list(args.test_years),
        "old_train_count": len(old_train),
        "old_train_class_counts": dict(sorted(old_labels.items())),
        "new_train_candidate_count": len(new_train_candidates),
        "new_train_count": len(new_train_rows),
        "combined_train_count": len(training_rows),
        "combined_train_class_counts": dict(sorted(training_labels.items())),
        "new_feature_row_count": len(new_rows),
        "merged_transfer_row_count": len(transfer_rows),
        "exact_duplicate_excluded_count": len(excluded_exact),
        "split_summaries": split_summaries,
        "outputs": {
            "features": str(merged_path),
            "split_manifest": str(manifest_path),
            "exact_duplicate_exclusions": str(excluded_path),
        },
        "blocking_reasons": [
            "near_duplicate_and_package_lineage_cross_source_audit_not_applied",
            "new_2021_malware_is_missing_and_is_not_in_primary_test_years",
            "this_is_a_cross_source_transfer_pilot_not_a_formal_confirmatory_result",
        ],
        "notes": [
            "Old-source rows from the fixed training year always enter training.",
            "When enabled, new-source rows from the same training year are added to training.",
            "New-source rows are never used for feature filtering or model fitting.",
            "Exact SHA256 collisions with the old training partition are excluded from test.",
            "Use the fixed-threshold compact evaluator after this script.",
        ],
    }
    audit_path = output / "transfer_split_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
