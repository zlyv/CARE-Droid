#!/usr/bin/env python3
"""Create grouped rolling-origin development splits inside fixed-train 2017.

Only APKs already assigned to the fixed 2017 training partition are read.
The provisional dex_date order is used, blocking groups remain intact, and
groups crossing a temporal boundary are excluded from that fold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.expanduser().resolve(strict=True).open(
        "r", encoding="utf-8-sig", errors="replace", newline=""
    ) as handle:
        return [
            {str(key).lstrip("\ufeff"): value for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_datetime(value: str) -> datetime:
    text = str(value or "").strip().replace("Z", "+00:00")
    if not text:
        raise ValueError("empty date")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        parsed = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def quantile_date(dates: list[datetime], fraction: float) -> datetime:
    index = min(len(dates) - 1, max(0, int(round(fraction * len(dates))) - 1))
    return dates[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--date-column", default="dex_date")
    parser.add_argument(
        "--train-end-fractions",
        nargs="+",
        type=float,
        default=[0.40, 0.60, 0.80],
    )
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    args = parser.parse_args()

    fractions = sorted(set(args.train_end_fractions))
    if not fractions or any(value <= 0.1 or value >= 0.9 for value in fractions):
        raise SystemExit("train-end fractions must be unique values between 0.1 and 0.9")
    if not 0.05 <= args.validation_fraction <= 0.40:
        raise SystemExit("validation fraction must be between 0.05 and 0.40")
    if any(value + args.validation_fraction > 1.000001 for value in fractions):
        raise SystemExit("train-end fraction plus validation fraction exceeds 1.0")

    split_path = args.split_manifest.expanduser().resolve(strict=True)
    metadata_path = args.metadata.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    split_rows = read_csv(split_path)
    metadata_rows = read_csv(metadata_path)
    source_split = "train_fixed_2017__test_2018"
    source_train = [
        row
        for row in split_rows
        if row.get("split_name") == source_split
        and row.get("partition") == "train"
    ]
    if not source_train:
        raise SystemExit(f"no fixed-2017 rows found in {source_split}")
    metadata_by_sha = {
        str(row.get("sha256", "")).strip().casefold(): row
        for row in metadata_rows
        if row.get("sha256")
    }

    enriched = []
    for row in source_train:
        sha256 = str(row.get("sha256", "")).strip().casefold()
        metadata = metadata_by_sha.get(sha256)
        if metadata is None:
            raise SystemExit(f"metadata row missing for {sha256}")
        try:
            date_value = parse_datetime(metadata.get(args.date_column, ""))
        except ValueError as exc:
            raise SystemExit(f"invalid {args.date_column} for {sha256}: {exc}") from exc
        enriched.append(
            {
                "sha256": sha256,
                "year": row.get("year", "2017"),
                "label": row.get("label", row.get("metadata_label", "")),
                "blocking_group": row.get("blocking_group") or sha256,
                "date": date_value,
            }
        )
    dates = sorted(row["date"] for row in enriched)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in enriched:
        groups[row["blocking_group"]].append(row)
    group_min = {
        group: min(row["date"] for row in rows) for group, rows in groups.items()
    }
    group_max = {
        group: max(row["date"] for row in rows) for group, rows in groups.items()
    }

    manifest_rows = []
    fold_summaries = []
    for fold_index, train_fraction in enumerate(fractions, start=1):
        validation_end_fraction = min(1.0, train_fraction + args.validation_fraction)
        train_cutoff = quantile_date(dates, train_fraction)
        validation_end = quantile_date(dates, validation_end_fraction)
        train_groups = {
            group for group in groups if group_max[group] <= train_cutoff
        }
        validation_groups = {
            group
            for group in groups
            if group_min[group] > train_cutoff
            and group_max[group] <= validation_end
        }
        overlap = train_groups & validation_groups
        if overlap:
            raise SystemExit(f"blocking-group overlap in fold {fold_index}")
        train_rows = [
            row for row in enriched if row["blocking_group"] in train_groups
        ]
        validation_rows = [
            row for row in enriched if row["blocking_group"] in validation_groups
        ]
        train_labels = Counter(row["label"] for row in train_rows)
        validation_labels = Counter(row["label"] for row in validation_rows)
        if len(train_labels) < 2 or len(validation_labels) < 2:
            raise SystemExit(
                f"single-class rolling fold {fold_index}: "
                f"train={dict(train_labels)} validation={dict(validation_labels)}"
            )
        if max(row["date"] for row in train_rows) >= min(
            row["date"] for row in validation_rows
        ):
            raise SystemExit(f"strict date order failed in fold {fold_index}")
        train_pct = int(round(train_fraction * 100))
        validation_pct = int(round(args.validation_fraction * 100))
        split_name = (
            f"rolling_2017_{fold_index}_train_through_{train_pct}pct"
            f"__validate_next_{validation_pct}pct"
        )
        for partition, rows in (("train", train_rows), ("test", validation_rows)):
            for row in sorted(rows, key=lambda item: (item["date"], item["sha256"])):
                manifest_rows.append(
                    {
                        "split_name": split_name,
                        "partition": partition,
                        "sha256": row["sha256"],
                        "year": row["year"],
                        "label": row["label"],
                        "blocking_group": row["blocking_group"],
                        "exclusion_reason": "",
                    }
                )
        used_groups = train_groups | validation_groups
        fold_summaries.append(
            {
                "split_name": split_name,
                "train_fraction_target": train_fraction,
                "validation_fraction_target": args.validation_fraction,
                "train_count": len(train_rows),
                "validation_count": len(validation_rows),
                "train_class_counts": dict(train_labels),
                "validation_class_counts": dict(validation_labels),
                "train_group_count": len(train_groups),
                "validation_group_count": len(validation_groups),
                "boundary_or_future_group_count": len(groups) - len(used_groups),
                "train_date_max": max(row["date"] for row in train_rows).isoformat(),
                "validation_date_min": min(
                    row["date"] for row in validation_rows
                ).isoformat(),
                "validation_date_max": max(
                    row["date"] for row in validation_rows
                ).isoformat(),
                "strict_date_order_verified": True,
                "blocking_group_overlap_count": 0,
            }
        )

    manifest_path = output / "split_manifest.csv"
    fields = [
        "split_name",
        "partition",
        "sha256",
        "year",
        "label",
        "blocking_group",
        "exclusion_reason",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P1_PROVISIONAL_2017_GROUPED_ROLLING_TEMPORAL_SPLITS",
        "formal_training_enabled": False,
        "protocol_status": "PROVISIONAL_DEX_DATE_NOT_AUDITED_FIRST_SEEN",
        "source_split_manifest": str(split_path),
        "source_split_manifest_sha256": sha256_file(split_path),
        "metadata_csv": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "date_column": args.date_column,
        "source_2017_row_count": len(enriched),
        "train_end_fractions": fractions,
        "validation_fraction": args.validation_fraction,
        "fold_summaries": fold_summaries,
        "outputs": {
            "split_manifest": str(manifest_path),
            "split_manifest_sha256": sha256_file(manifest_path),
        },
        "blocking_reasons": [
            "dex_date_is_not_audited_first_seen_utc",
            "rolling_splits_are_for_development_only",
        ],
        "notes": [
            "Only fixed-2017 training rows are read.",
            "2018--2021 rows are never read or used.",
            "Blocking groups remain intact inside every fold.",
            "Groups crossing a fold boundary are excluded from that fold.",
        ],
    }
    audit_path = output / "rolling_temporal_split_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
