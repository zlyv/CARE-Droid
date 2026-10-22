#!/usr/bin/env python3
"""Create a provisional chronological development split inside 2017.

This is only for pilot hyperparameter selection.  It uses the metadata date
field (normally dex_date), keeps blocking groups intact, assigns the latest
groups to validation, and never reads 2018--2021 rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def read_csv(path: Path):
    with path.expanduser().resolve(strict=True).open(
        "r", encoding="utf-8-sig", errors="replace", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append(
                {
                    str(key).lstrip("\ufeff"): value
                    for key, value in row.items()
                }
            )
        return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_datetime(value: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("empty date")
    text = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        parsed = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def require_column(rows, name, description):
    if not rows or name not in rows[0]:
        available = sorted(rows[0]) if rows else []
        raise SystemExit(
            f"{description} column not found: {name}; available={available}"
        )
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split-name", default="train_2017_early__validate_2017_late")
    parser.add_argument("--date-column", default="dex_date")
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    args = parser.parse_args()
    if not 0.05 <= args.validation_fraction <= 0.45:
        raise SystemExit("validation fraction must be between 0.05 and 0.45")

    split_path = args.split_manifest.expanduser().resolve(strict=True)
    metadata_path = args.metadata.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    split_rows = read_csv(split_path)
    metadata_rows = read_csv(metadata_path)
    if not split_rows:
        raise SystemExit("split manifest has no rows")
    if not metadata_rows:
        raise SystemExit("metadata CSV has no rows")
    require_column(metadata_rows, args.date_column, "date")
    require_column(metadata_rows, "sha256", "metadata SHA256")
    require_column(split_rows, "split_name", "split name")
    require_column(split_rows, "partition", "partition")
    require_column(split_rows, "sha256", "split SHA256")
    require_column(split_rows, "blocking_group", "blocking group")

    source_split = "train_fixed_2017__test_2018"
    train_rows = [
        row for row in split_rows
        if row.get("split_name") == source_split
        and row.get("partition") == "train"
    ]
    if not train_rows:
        raise SystemExit(f"no 2017 training rows found in {source_split}")

    metadata_by_sha = {
        str(row.get("sha256", "")).strip().lower(): row
        for row in metadata_rows
        if row.get("sha256")
    }
    enriched = []
    missing = []
    invalid = []
    for row in train_rows:
        sha = str(row["sha256"]).strip().lower()
        metadata = metadata_by_sha.get(sha)
        if metadata is None:
            missing.append(sha)
            continue
        try:
            date_value = parse_datetime(metadata.get(args.date_column, ""))
        except ValueError:
            invalid.append(sha)
            continue
        enriched.append(
            {
                "sha256": sha,
                "year": row.get("year", "2017"),
                "label": row.get("label", row.get("metadata_label", "")),
                "blocking_group": row["blocking_group"],
                "date": date_value,
            }
        )
    if missing or invalid:
        raise SystemExit(
            f"metadata date join failed: missing={len(missing)} invalid={len(invalid)}"
        )

    groups = defaultdict(list)
    for row in enriched:
        groups[row["blocking_group"]].append(row)
    group_min = {
        group: min(row["date"] for row in rows)
        for group, rows in groups.items()
    }
    group_max = {
        group: max(row["date"] for row in rows)
        for group, rows in groups.items()
    }
    target_validation = max(1, int(round(len(enriched) * args.validation_fraction)))
    candidate_dates = sorted({row["date"] for row in enriched})
    best = None
    for cutoff in candidate_dates[:-1]:
        train_groups = {
            group for group in groups if group_max[group] <= cutoff
        }
        validation_groups = {
            group for group in groups if group_min[group] > cutoff
        }
        early_rows_candidate = [
            row for row in enriched if row["blocking_group"] in train_groups
        ]
        validation_rows_candidate = [
            row for row in enriched if row["blocking_group"] in validation_groups
        ]
        if not early_rows_candidate or not validation_rows_candidate:
            continue
        early_labels_candidate = Counter(
            row["label"] for row in early_rows_candidate
        )
        validation_labels_candidate = Counter(
            row["label"] for row in validation_rows_candidate
        )
        if len(early_labels_candidate) < 2 or len(validation_labels_candidate) < 2:
            continue
        used_count = len(early_rows_candidate) + len(validation_rows_candidate)
        validation_fraction = len(validation_rows_candidate) / used_count
        score = (
            abs(validation_fraction - args.validation_fraction),
            -used_count,
            cutoff,
        )
        if best is None or score < best[0]:
            best = (
                score,
                cutoff,
                train_groups,
                validation_groups,
                early_rows_candidate,
                validation_rows_candidate,
            )
    if best is None:
        raise SystemExit(
            "could not form a strict chronological split with both classes "
            "while keeping blocking groups intact"
        )
    (
        _score,
        cutoff,
        train_groups,
        validation_groups,
        early_rows,
        validation_rows,
    ) = best
    excluded_rows = [
        row for row in enriched
        if row["blocking_group"] not in train_groups
        and row["blocking_group"] not in validation_groups
    ]
    if not early_rows or not validation_rows:
        raise SystemExit("chronological split produced an empty partition")
    early_labels = Counter(row["label"] for row in early_rows)
    validation_labels = Counter(row["label"] for row in validation_rows)
    if len(early_labels) < 2 or len(validation_labels) < 2:
        raise SystemExit(
            f"chronological split is single-class: train={dict(early_labels)} "
            f"validation={dict(validation_labels)}"
        )

    manifest_rows = []
    for partition, rows in (("train", early_rows), ("test", validation_rows)):
        for row in sorted(rows, key=lambda item: (item["date"], item["sha256"])):
            manifest_rows.append(
                {
                    "split_name": args.split_name,
                    "partition": partition,
                    "sha256": row["sha256"],
                    "year": row["year"],
                    "label": row["label"],
                    "blocking_group": row["blocking_group"],
                    "exclusion_reason": "",
                }
            )
    manifest_path = output / "split_manifest.csv"
    fields = [
        "split_name", "partition", "sha256", "year", "label",
        "blocking_group", "exclusion_reason",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P1_PROVISIONAL_2017_CHRONOLOGICAL_DEVELOPMENT_SPLIT",
        "formal_training_enabled": False,
        "protocol_status": "PROVISIONAL_DEX_DATE_NOT_AUDITED_FIRST_SEEN",
        "source_split_manifest": str(split_path),
        "source_split_manifest_sha256": sha256_file(split_path),
        "metadata_csv": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "date_column": args.date_column,
        "split_name": args.split_name,
        "validation_fraction_target": args.validation_fraction,
        "row_count": len(enriched),
        "train_count": len(early_rows),
        "validation_count": len(validation_rows),
        "excluded_count": len(excluded_rows),
        "train_class_counts": dict(early_labels),
        "validation_class_counts": dict(validation_labels),
        "train_date_max": max(row["date"] for row in early_rows).isoformat(),
        "validation_date_min": min(row["date"] for row in validation_rows).isoformat(),
        "cutoff_date": cutoff.isoformat(),
        "strict_date_order_verified": (
            max(row["date"] for row in early_rows)
            < min(row["date"] for row in validation_rows)
        ),
        "blocking_group_count": len(groups),
        "train_blocking_group_count": len(train_groups),
        "validation_blocking_group_count": len(validation_groups),
        "excluded_blocking_group_count": len(groups) - len(train_groups) - len(validation_groups),
        "outputs": {
            "split_manifest": str(manifest_path),
            "split_manifest_sha256": sha256_file(manifest_path),
        },
        "blocking_reasons": [
            "dex_date_is_not_audited_first_seen_utc",
            "this_split_is_for_pilot_hyperparameter_selection_only",
        ],
        "notes": [
            "Only rows already assigned to the fixed-2017 training partition are used.",
            "Blocking groups are kept intact and the latest groups form validation.",
            "2018--2021 rows are not read or used.",
        ],
    }
    audit_path = output / "temporal_inner_split_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
