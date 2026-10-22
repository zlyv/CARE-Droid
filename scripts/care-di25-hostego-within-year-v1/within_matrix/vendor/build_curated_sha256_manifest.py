#!/usr/bin/env python3
"""Build a normalized metadata CSV from curated SHA256 list files.

The new non-packed source currently provides curated ``*.sha256.txt`` files
whose filenames encode label and year.  This script converts those lists into
the CSV shape consumed by the metadata join audit.  It does not inspect APK
contents and deliberately leaves APK size, package name, dex date, and scan
date empty.  The provenance is recorded explicitly as
``CURATED_SHA256_LIST_FILENAME``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


FILE_RE = re.compile(
    r"^(?P<label>benign|malware|malicious|clean|negative)_"
    r"(?P<year>20[0-9]{2})(?:_|\.)",
    re.IGNORECASE,
)
SHA_RE = re.compile(r"(?i)^[0-9a-f]{64}$")
LABEL_MAP = {
    "benign": ("0", "benign"),
    "clean": ("0", "benign"),
    "negative": ("0", "benign"),
    "malware": ("1", "malicious"),
    "malicious": ("1", "malicious"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_file_name(path: Path) -> tuple[str, int]:
    match = FILE_RE.match(path.name)
    if not match:
        raise ValueError(f"cannot infer label/year from filename: {path.name}")
    label = match.group("label").casefold()
    return label, int(match.group("year"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-id", default="new_nonpacked_2017_2021")
    args = parser.parse_args()

    list_root = args.list_root.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows_by_sha: dict[str, dict] = {}
    conflicts = []
    file_reports = []

    for path in sorted(list_root.rglob("*.sha256.txt")):
        label_key, year = parse_file_name(path)
        label_value, label_name = LABEL_MAP[label_key]
        row_count = 0
        invalid_count = 0
        duplicate_count = 0
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            value = raw.strip().split()[0] if raw.strip() else ""
            if not value:
                continue
            if not SHA_RE.fullmatch(value):
                invalid_count += 1
                continue
            digest = value.casefold()
            row_count += 1
            candidate = {
                "source_id": args.source_id,
                "sha256": digest,
                "year": year,
                "label": label_value,
                "label_name": label_name,
                "apk_size": "",
                "dex_date": "",
                "vt_scan_date": "",
                "markets": "",
                "pkg_name": "",
                "metadata_provenance": "CURATED_SHA256_LIST_FILENAME",
                "metadata_source_file": str(path),
            }
            previous = rows_by_sha.get(digest)
            if previous is not None:
                duplicate_count += 1
                if (previous["year"], previous["label"]) != (year, label_value):
                    conflicts.append(
                        {
                            "sha256": digest,
                            "first_year": previous["year"],
                            "first_label": previous["label"],
                            "second_year": year,
                            "second_label": label_value,
                            "first_source_file": previous["metadata_source_file"],
                            "second_source_file": str(path),
                        }
                    )
                continue
            rows_by_sha[digest] = candidate
        file_reports.append(
            {
                "file": str(path),
                "year": year,
                "label": label_name,
                "row_count": row_count,
                "invalid_count": invalid_count,
                "duplicate_count": duplicate_count,
                "sha256": sha256_file(path),
            }
        )

    if conflicts:
        raise SystemExit(
            f"found {len(conflicts)} cross-label/year SHA conflicts; inspect the source lists"
        )
    if not rows_by_sha:
        raise SystemExit("no valid SHA256 rows found")

    fields = [
        "source_id", "sha256", "year", "label", "label_name", "apk_size",
        "dex_date", "vt_scan_date", "markets", "pkg_name",
        "metadata_provenance", "metadata_source_file",
    ]
    csv_path = output / "curated_sha256_metadata.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_by_sha[digest] for digest in sorted(rows_by_sha))

    manifest_by_year_label = Counter(
        (row["year"], row["label_name"]) for row in rows_by_sha.values()
    )
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "FORMAL_NEW_SOURCE_CURATED_METADATA_MANIFEST",
        "source_id": args.source_id,
        "list_root": str(list_root),
        "list_file_count": len(file_reports),
        "metadata_row_count": len(rows_by_sha),
        "metadata_unique_sha256_count": len(rows_by_sha),
        "counts_by_year_and_label": [
            {"year": year, "label": label, "count": count}
            for (year, label), count in sorted(manifest_by_year_label.items())
        ],
        "file_reports": file_reports,
        "outputs": {
            "metadata_csv": str(csv_path),
            "conflicts_csv": str(output / "curated_metadata_conflicts.csv"),
        },
        "provenance": "CURATED_SHA256_LIST_FILENAME",
        "blocking_reasons": [
            "apk_inventory_join_coverage_not_yet_checked",
            "first_seen_utc_not_available_from_sha256_lists",
            "package_certificate_version_metadata_not_available_from_sha256_lists",
        ],
        "notes": [
            "Year and label come from curated list filenames, not APK content parsing.",
            "APK size, package name, dex date, and VT scan date are intentionally blank.",
            "Do not proceed to formal training until inventory-to-manifest coverage is audited.",
        ],
    }
    conflict_path = output / "curated_metadata_conflicts.csv"
    with conflict_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sha256", "first_year", "first_label", "second_year",
                "second_label", "first_source_file", "second_source_file",
            ],
        )
        writer.writeheader()
        writer.writerows(conflicts)
    (output / "curated_metadata_manifest_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
