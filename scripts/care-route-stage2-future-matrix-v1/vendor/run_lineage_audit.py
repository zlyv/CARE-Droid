#!/usr/bin/env python3
"""Build package/certificate/version-line audit tables from P0 CSV outputs."""

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
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def norm(value):
    return str(value or "").strip()


def version_number(value):
    text = norm(value)
    try:
        return int(text)
    except ValueError:
        return -1


def main():
    parser = argparse.ArgumentParser(description="Audit APK package/certificate version lines")
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    identity_rows = read_csv(args.identity)
    metadata_rows = read_csv(args.metadata)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    metadata_by_sha = {norm(row.get("sha256")).lower(): row for row in metadata_rows}
    eligible = []
    excluded = []
    for row in identity_rows:
        digest = norm(row.get("sha256")).lower()
        package = norm(row.get("package_name"))
        certificate = norm(row.get("certificate_sha256"))
        if row.get("manifest_status") != "OK" or row.get("certificate_status") != "OK" or not package or not certificate:
            excluded.append({
                "sha256": digest,
                "absolute_path": row.get("absolute_path", ""),
                "package_name": package,
                "certificate_status": row.get("certificate_status", ""),
                "reason": "IDENTITY_UNRESOLVED",
            })
            continue
        metadata = metadata_by_sha.get(digest, {})
        eligible.append({
            "sha256": digest,
            "absolute_path": row.get("absolute_path", ""),
            "source_id": row.get("source_id", ""),
            "package_name": package,
            "certificate_sha256": certificate,
            "version_code": norm(row.get("version_code")),
            "version_name": norm(row.get("version_name")),
            "min_sdk": norm(row.get("min_sdk")),
            "target_sdk": norm(row.get("target_sdk")),
            "metadata_year": norm(metadata.get("metadata_year")),
            "metadata_label": norm(metadata.get("metadata_label")),
            "dex_date": norm(metadata.get("dex_date")),
            "metadata_pkg_name": norm(metadata.get("metadata_pkg_name")),
        })

    group_members = defaultdict(list)
    package_certs = defaultdict(set)
    package_versions = defaultdict(set)
    version_keys = defaultdict(list)
    for row in eligible:
        key = f"{row['package_name']}|{row['certificate_sha256']}"
        group_members[key].append(row)
        package_certs[row["package_name"]].add(row["certificate_sha256"])
        package_versions[(row["package_name"], row["version_code"])].add(row["certificate_sha256"])
        version_keys[(row["package_name"], row["certificate_sha256"], row["version_code"])].append(row)

    group_ids = {
        key: f"PCV{index:06d}"
        for index, key in enumerate(sorted(group_members), start=1)
    }
    lineage_fields = [
        "lineage_id", "lineage_key", "sha256", "absolute_path", "source_id",
        "package_name", "certificate_sha256", "version_code", "version_name",
        "min_sdk", "target_sdk", "metadata_year", "metadata_label", "dex_date",
        "package_certificate_count", "same_version_other_certificate_count",
    ]
    lineage_rows = []
    for key, members in sorted(group_members.items()):
        package, certificate = key.split("|", 1)
        same_package_versions = {
            (row["package_name"], row["version_code"]): row["certificate_sha256"]
            for row in members
        }
        for row in sorted(
            members,
            key=lambda value: (version_number(value["version_code"]), value["version_name"], value["sha256"]),
        ):
            other_certs = package_versions[(package, row["version_code"])] - {certificate}
            lineage_rows.append({
                "lineage_id": group_ids[key],
                "lineage_key": key,
                **row,
                "package_certificate_count": len(package_certs[package]),
                "same_version_other_certificate_count": len(other_certs),
            })

    write_csv(output / "package_certificate_lineage.csv", lineage_rows, lineage_fields)
    write_csv(
        output / "identity_excluded_from_lineage.csv",
        excluded,
        ["sha256", "absolute_path", "package_name", "certificate_status", "reason"],
    )

    conflicts = []
    for (package, version_code), certificates in sorted(package_versions.items()):
        if len(certificates) > 1:
            conflicts.append({
                "conflict_type": "SAME_PACKAGE_VERSION_DIFFERENT_CERTIFICATE",
                "package_name": package,
                "version_code": version_code,
                "certificate_count": len(certificates),
                "certificates": "|".join(sorted(certificates)),
            })
    for (package, certificate, version_code), members in sorted(version_keys.items()):
        if len(members) > 1:
            conflicts.append({
                "conflict_type": "SAME_PACKAGE_CERT_VERSION_MULTIPLE_APKS",
                "package_name": package,
                "version_code": version_code,
                "certificate_count": 1,
                "certificates": certificate,
            })
    write_csv(
        output / "lineage_conflicts.csv",
        conflicts,
        ["conflict_type", "package_name", "version_code", "certificate_count", "certificates"],
    )

    package_count = len({row["package_name"] for row in eligible})
    certificate_count = len({row["certificate_sha256"] for row in eligible})
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P0_PACKAGE_CERTIFICATE_VERSION_LINE_AUDIT",
        "formal_training_enabled": False,
        "identity_csv": str(args.identity.expanduser().resolve()),
        "metadata_csv": str(args.metadata.expanduser().resolve()),
        "identity_row_count": len(identity_rows),
        "identity_resolved_count": len(eligible),
        "identity_excluded_count": len(excluded),
        "unique_package_count": package_count,
        "unique_certificate_count": certificate_count,
        "package_certificate_line_count": len(group_members),
        "lineage_conflict_count": len(conflicts),
        "package_with_multiple_certificates_count": sum(len(certs) > 1 for certs in package_certs.values()),
        "blocking_reasons": [
            "first_seen_utc_not_joined",
            "near_duplicate_clusters_not_built",
            "static_parse_and_api_graph_coverage_not_available",
            "resolved_formal_training_config_not_generated",
        ],
        "notes": [
            "lineage key is exact package_name plus exact certificate_sha256",
            "identity-unresolved APKs are retained in an explicit exclusion file",
            "same-package certificate rotations are reported as conflicts, not silently merged",
            "call graphs are not built or persisted in this phase",
        ],
    }
    (output / "lineage_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
