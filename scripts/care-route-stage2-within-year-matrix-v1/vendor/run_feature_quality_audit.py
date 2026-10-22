#!/usr/bin/env python3
"""Audit compact features and write a pilot-only model feature whitelist.

This script never trains a model.  It separates audit/identity fields from
numeric APK features, reports missing/non-numeric/constant columns, and writes
a whitelist for pilot diagnostics.  The whitelist is explicitly not a formal
training decision: formal feature filtering must be recomputed inside each
training fold after the final split protocol is locked.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = "1.0.0"

# These fields are metadata, identity, labels, paths, or parser diagnostics.
# They must never enter the classifier matrix.
EXCLUDED_AUDIT_COLUMNS = {
    "sha256",
    "absolute_path",
    "source_id",
    "lineage_id",
    "metadata_year",
    "metadata_label",
    "package_name",
    "certificate_sha256",
    "version_code",
    "status",
    "error_type",
    "error_message",
}

MISSING_VALUES = {"", "na", "nan", "none", "null"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_missing(value: str) -> bool:
    return str(value or "").strip().lower() in MISSING_VALUES


def parse_number(value: str):
    if is_missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def read_csv(path: Path):
    with path.expanduser().resolve(strict=True).open(
        "r", encoding="utf-8", errors="replace", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--schema", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    features_path = args.features.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows = read_csv(features_path)
    if not rows:
        raise SystemExit("feature CSV has no data rows")
    header = list(rows[0].keys())
    missing_required = sorted({"sha256", "metadata_year", "metadata_label"} - set(header))
    if missing_required:
        raise SystemExit(f"feature CSV is missing required columns: {missing_required}")

    duplicate_sha = sum(
        count - 1 for count in Counter(row.get("sha256", "") for row in rows).values()
        if count > 1
    )
    candidate_columns = [name for name in header if name not in EXCLUDED_AUDIT_COLUMNS]
    reports = []
    numeric_nonconstant = []
    constant_columns = []
    nonnumeric_columns = []
    missing_columns = []

    for name in candidate_columns:
        raw_values = [row.get(name, "") for row in rows]
        numbers = [parse_number(value) for value in raw_values]
        missing_count = sum(value is None for value in numbers)
        nonnumeric_count = sum(
            not is_missing(value) and number is None
            for value, number in zip(raw_values, numbers)
        )
        valid_numbers = [number for number in numbers if number is not None]
        unique_values = {number for number in valid_numbers}
        is_numeric = nonnumeric_count == 0
        is_constant = is_numeric and len(unique_values) <= 1
        if missing_count:
            missing_columns.append(name)
        if not is_numeric:
            nonnumeric_columns.append(name)
        if is_constant:
            constant_columns.append(name)
        if is_numeric and not is_constant and not missing_count:
            numeric_nonconstant.append(name)

        reports.append(
            {
                "feature": name,
                "is_numeric": is_numeric,
                "missing_count": missing_count,
                "nonnumeric_count": nonnumeric_count,
                "unique_numeric_count": len(unique_values),
                "nonzero_count": sum(number != 0 for number in valid_numbers),
                "min": min(valid_numbers) if valid_numbers else None,
                "max": max(valid_numbers) if valid_numbers else None,
                "constant": is_constant,
                "pilot_whitelist": name in numeric_nonconstant,
            }
        )

    unknown_labels = sorted(
        {
            str(row.get("metadata_label", "")).strip().lower()
            for row in rows
            if str(row.get("metadata_label", "")).strip().lower()
            not in {"benign", "malware", "malicious"}
        }
    )
    years = Counter(str(row.get("metadata_year", "")) for row in rows)
    status_counts = Counter(str(row.get("status", "")) for row in rows)

    report_path = output / "feature_quality.csv"
    write_csv(report_path, reports, list(reports[0].keys()))

    schema_sha = None
    schema_data = None
    if args.schema:
        schema_path = args.schema.expanduser().resolve(strict=True)
        schema_sha = sha256_file(schema_path)
        schema_data = json.loads(schema_path.read_text(encoding="utf-8"))

    whitelist = {
        "schema_version": SCHEMA_VERSION,
        "scope": "PILOT_DIAGNOSTIC_ONLY",
        "formal_training_eligible": False,
        "source_features_sha256": sha256_file(features_path),
        "source_schema_sha256": schema_sha,
        "excluded_audit_columns": sorted(EXCLUDED_AUDIT_COLUMNS),
        "candidate_feature_count": len(candidate_columns),
        "pilot_numeric_nonconstant_features": numeric_nonconstant,
        "constant_features": sorted(constant_columns),
        "nonnumeric_features": sorted(nonnumeric_columns),
        "missing_value_features": sorted(missing_columns),
        "notes": [
            "This whitelist is for pilot diagnostics only.",
            "Formal filtering must be recomputed inside each training fold.",
            "The semantic mapping schema is not modified by constant-column removal.",
        ],
    }
    (output / "pilot_feature_whitelist.json").write_text(
        json.dumps(whitelist, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    audit = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P1_FEATURE_QUALITY_AND_LEAKAGE_AUDIT",
        "formal_training_enabled": False,
        "features_csv": str(features_path),
        "features_sha256": sha256_file(features_path),
        "schema_json": str(args.schema.expanduser().resolve()) if args.schema else None,
        "schema_sha256": schema_sha,
        "row_count": len(rows),
        "column_count": len(header),
        "duplicate_sha_extra_rows": duplicate_sha,
        "candidate_feature_count": len(candidate_columns),
        "numeric_nonconstant_feature_count": len(numeric_nonconstant),
        "constant_feature_count": len(constant_columns),
        "nonnumeric_feature_count": len(nonnumeric_columns),
        "missing_value_feature_count": len(missing_columns),
        "constant_features": sorted(constant_columns),
        "nonnumeric_features": sorted(nonnumeric_columns),
        "missing_value_features": sorted(missing_columns),
        "unknown_labels": unknown_labels,
        "year_counts": dict(sorted(years.items())),
        "status_counts": dict(status_counts),
        "excluded_audit_columns": sorted(EXCLUDED_AUDIT_COLUMNS),
        "outputs": {
            "quality_csv": str(report_path),
            "pilot_whitelist": str(output / "pilot_feature_whitelist.json"),
        },
        "blocking_reasons": [
            "audited_first_seen_utc_not_joined",
            "formal_training_config_not_generated",
            "formal_fold_local_feature_filter_not_resolved",
        ],
        "notes": [
            "No model is trained in this phase.",
            "Metadata, identity, path, label, and parser fields are excluded explicitly.",
            "Constant columns are reported and removed only from the pilot whitelist.",
        ],
    }
    (output / "feature_quality_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
