#!/usr/bin/env python3
"""Build provisional annual forward-time split manifests.

The current metadata has year but no audited first_seen_utc, so this is not
the formal quarter protocol.  It is a pilot manifest for comparing TIF and
TASA-Droid on the same 2018--2021 annual test years.  Exact/identity blocking
components are respected: a test row is excluded when its component already
appears in the preceding training years.

By default, each test year uses all preceding years for training.  With
``--fixed-train-years``, every requested test year uses exactly the declared
training years.  The latter is intended for a matched fixed-2017 protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = "1.0.0"
TEST_YEARS = (2018, 2019, 2020, 2021)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path):
    with path.expanduser().resolve(strict=True).open(
        "r", encoding="utf-8", errors="replace", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames:
            reader.fieldnames = [
                str(name).lstrip("\ufeff") if name is not None else name
                for name in reader.fieldnames
            ]
        rows = []
        for row in reader:
            rows.append(
                {
                    (str(key).lstrip("\ufeff") if key is not None else key): value
                    for key, value in row.items()
                }
            )
        return rows


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_column(rows, explicit, candidates, description):
    if explicit:
        if rows and explicit not in rows[0]:
            raise SystemExit(f"requested {description} column not found: {explicit}")
        return explicit
    if not rows:
        raise SystemExit(f"{description} CSV has no rows")
    for candidate in candidates:
        if candidate in rows[0]:
            return candidate
    raise SystemExit(
        f"could not infer {description} column; use the explicit option. "
        f"available columns: {sorted(rows[0])}"
    )


def label_value(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized == "benign":
        return "benign"
    if normalized in {"malware", "malicious"}:
        return "malicious"
    return normalized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--blocking", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--group-column")
    parser.add_argument("--test-years", nargs="+", type=int, default=list(TEST_YEARS))
    parser.add_argument(
        "--fixed-train-years",
        nargs="+",
        type=int,
        help="Use exactly these training years for every test year instead of all prior years.",
    )
    args = parser.parse_args()

    test_years = sorted(set(args.test_years))
    fixed_train_years = (
        sorted(set(args.fixed_train_years)) if args.fixed_train_years else None
    )
    if fixed_train_years is not None:
        invalid = [year for year in fixed_train_years if year >= min(test_years)]
        if invalid:
            raise SystemExit(
                "fixed training years must be earlier than every requested test year; "
                f"invalid={invalid}, test_years={test_years}"
            )

    feature_path = args.features.expanduser().resolve(strict=True)
    blocking_path = args.blocking.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    feature_rows = read_csv(feature_path)
    blocking_rows = read_csv(blocking_path)
    if not feature_rows:
        raise SystemExit("feature CSV has no rows")
    if not blocking_rows:
        raise SystemExit("blocking CSV has no rows")

    feature_sha_column = find_column(
        feature_rows, "sha256", ("sha256", "SHA256"), "feature SHA256"
    )
    blocking_sha_column = find_column(
        blocking_rows, None, ("sha256", "SHA256"), "blocking SHA256"
    )
    group_column = find_column(
        blocking_rows,
        args.group_column,
        (
            "blocking_component_id",
            "component_id",
            "blocking_id",
            "block_id",
            "cluster_id",
            "group_id",
        ),
        "blocking group",
    )

    groups = {}
    for row in blocking_rows:
        sha = str(row.get(blocking_sha_column, "")).strip().lower()
        group = str(row.get(group_column, "")).strip()
        if not sha or not group:
            continue
        previous = groups.get(sha)
        if previous is not None and previous != group:
            raise SystemExit(f"blocking SHA has conflicting groups: {sha}")
        groups[sha] = group

    normalized = []
    missing_group = []
    unknown_label = []
    invalid_year = []
    for row in feature_rows:
        sha = str(row.get(feature_sha_column, "")).strip().lower()
        label = label_value(row.get("metadata_label", ""))
        try:
            year = int(str(row.get("metadata_year", "")).strip())
        except ValueError:
            year = -1
        group = groups.get(sha)
        if not group:
            missing_group.append(sha)
        if label not in {"benign", "malicious"}:
            unknown_label.append((sha, label))
        if year < 0:
            invalid_year.append((sha, row.get("metadata_year", "")))
        normalized.append({"sha256": sha, "label": label, "year": year, "group": group or ""})

    if missing_group:
        sample = missing_group[:10]
        raise SystemExit(
            f"{len(missing_group)} feature rows are missing blocking groups; sample={sample}"
        )
    if unknown_label:
        raise SystemExit(f"unknown labels found; sample={unknown_label[:10]}")
    if invalid_year:
        raise SystemExit(f"invalid metadata years found; sample={invalid_year[:10]}")

    manifest_rows = []
    summaries = []
    for test_year in test_years:
        if fixed_train_years is None:
            train_items = [item for item in normalized if item["year"] < test_year]
            split_name = f"train_before_{test_year}__test_{test_year}"
        else:
            train_items = [item for item in normalized if item["year"] in fixed_train_years]
            split_name = (
                f"train_fixed_{'-'.join(str(year) for year in fixed_train_years)}"
                f"__test_{test_year}"
            )
        test_items = [item for item in normalized if item["year"] == test_year]
        prior_groups = {item["group"] for item in train_items}
        test_excluded = [item for item in test_items if item["group"] in prior_groups]
        test_kept = [item for item in test_items if item["group"] not in prior_groups]

        for item in train_items:
            manifest_rows.append(
                {
                    "split_name": split_name,
                    "partition": "train",
                    "sha256": item["sha256"],
                    "year": item["year"],
                    "label": item["label"],
                    "blocking_group": item["group"],
                    "exclusion_reason": "",
                }
            )
        for item in test_kept:
            manifest_rows.append(
                {
                    "split_name": split_name,
                    "partition": "test",
                    "sha256": item["sha256"],
                    "year": item["year"],
                    "label": item["label"],
                    "blocking_group": item["group"],
                    "exclusion_reason": "",
                }
            )
        for item in test_excluded:
            manifest_rows.append(
                {
                    "split_name": split_name,
                    "partition": "excluded",
                    "sha256": item["sha256"],
                    "year": item["year"],
                    "label": item["label"],
                    "blocking_group": item["group"],
                    "exclusion_reason": "blocking_group_seen_in_training_years",
                }
            )

        summaries.append(
            {
                "split_name": split_name,
                "test_year": test_year,
                "train_count": len(train_items),
                "test_count": len(test_kept),
                "test_excluded_count": len(test_excluded),
                "train_class_counts": dict(Counter(item["label"] for item in train_items)),
                "test_class_counts": dict(Counter(item["label"] for item in test_kept)),
                "excluded_class_counts": dict(Counter(item["label"] for item in test_excluded)),
            }
        )

    csv_path = output / "split_manifest.csv"
    fields = [
        "split_name", "partition", "sha256", "year", "label",
        "blocking_group", "exclusion_reason",
    ]
    write_csv(csv_path, manifest_rows, fields)
    manifest_sha = sha256_file(csv_path)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": (
            "P1_PROVISIONAL_FIXED_SOURCE_YEAR_SPLIT"
            if fixed_train_years is not None
            else "P1_PROVISIONAL_ANNUAL_FORWARD_SPLIT"
        ),
        "formal_training_enabled": False,
        "protocol_status": (
            "PROVISIONAL_FIXED_TRAIN_YEARS_NO_AUDITED_FIRST_SEEN"
            if fixed_train_years is not None
            else "PROVISIONAL_YEAR_ONLY_NO_AUDITED_FIRST_SEEN"
        ),
        "features_csv": str(feature_path),
        "features_sha256": sha256_file(feature_path),
        "blocking_csv": str(blocking_path),
        "blocking_sha256": sha256_file(blocking_path),
        "blocking_group_column": group_column,
        "test_years": test_years,
        "training_policy": "fixed_years" if fixed_train_years is not None else "all_prior_years",
        "fixed_train_years": fixed_train_years,
        "row_count": len(feature_rows),
        "manifest_row_count": len(manifest_rows),
        "split_summaries": summaries,
        "outputs": {
            "split_manifest_csv": str(csv_path),
            "split_manifest_sha256": manifest_sha,
        },
        "blocking_reasons": [
            "audited_first_seen_utc_not_joined",
            "calendar_quarter_split_not_available",
            "resolved_formal_training_config_not_generated",
        ],
        "notes": [
            "Annual metadata year is used only for the pilot feasibility split.",
            "A test row is excluded when its exact/identity blocking group was seen in the selected training years.",
            "Fixed training years are intended for protocol-matched baseline comparisons, not cumulative deployment claims.",
            "This manifest must be replaced by the audited first-seen quarter manifest before formal training.",
        ],
    }
    (output / "split_manifest_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
