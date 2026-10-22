#!/usr/bin/env python3
"""Fingerprint APKs involved in package/certificate/version collisions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def read_csv(path):
    with Path(path).expanduser().resolve(strict=True).open(
        "r", encoding="utf-8-sig", errors="replace", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def digest_entries(archive, names):
    digest = hashlib.sha256()
    for name in sorted(names):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(archive.read(name))
    return digest.hexdigest()


def fingerprint(row):
    path = Path(row["absolute_path"])
    result = {
        "sha256": row["sha256"],
        "absolute_path": str(path),
        "package_name": row["package_name"],
        "certificate_sha256": row["certificate_sha256"],
        "version_code": row["version_code"],
        "version_name": row["version_name"],
        "metadata_year": row.get("metadata_year", ""),
        "metadata_label": row.get("metadata_label", ""),
        "dex_entry_count": "",
        "dex_fingerprint": "",
        "manifest_fingerprint": "",
        "zip_entry_count": "",
        "fingerprint_status": "ERROR",
        "error_type": "",
        "error_message": "",
    }
    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = archive.namelist()
            dex_names = [name for name in names if name.lower().startswith("classes") and name.lower().endswith(".dex")]
            manifest_names = [name for name in names if name.lower() == "androidmanifest.xml"]
            result["zip_entry_count"] = len(names)
            result["dex_entry_count"] = len(dex_names)
            result["dex_fingerprint"] = digest_entries(archive, dex_names) if dex_names else "NO_DEX"
            result["manifest_fingerprint"] = digest_entries(archive, manifest_names) if manifest_names else "NO_MANIFEST"
            result["fingerprint_status"] = "OK"
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)[:500]
    return result


def main():
    parser = argparse.ArgumentParser(description="Fingerprint collision APKs")
    parser.add_argument("--lineage", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = read_csv(args.lineage)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    package_certs = defaultdict(set)
    collision_groups = defaultdict(list)
    for row in rows:
        package_certs[row["package_name"]].add(row["certificate_sha256"])
        key = (row["package_name"], row["certificate_sha256"], row["version_code"])
        collision_groups[key].append(row)

    selected = []
    selected_keys = set()
    for key, members in collision_groups.items():
        if len(members) > 1:
            selected_keys.add(key)
            selected.extend(members)
    for row in rows:
        if len(package_certs[row["package_name"]]) > 1:
            key = (row["package_name"], row["certificate_sha256"], row["version_code"])
            if key not in selected_keys:
                selected_keys.add(key)
                selected.append(row)

    fingerprints = [fingerprint(row) for row in selected]
    fields = [
        "sha256", "absolute_path", "package_name", "certificate_sha256", "version_code",
        "version_name", "metadata_year", "metadata_label", "dex_entry_count",
        "dex_fingerprint", "manifest_fingerprint", "zip_entry_count",
        "fingerprint_status", "error_type", "error_message",
    ]
    write_csv(output / "collision_fingerprints.csv", fingerprints, fields)

    dex_groups = defaultdict(list)
    for row in fingerprints:
        dex_groups[(row["package_name"], row["certificate_sha256"], row["version_code"])].append(row)
    group_rows = []
    for key, members in sorted(dex_groups.items()):
        is_multi_certificate_singleton = (
            len(members) == 1 and len(package_certs[key[0]]) > 1
        )
        if is_multi_certificate_singleton:
            classification = "MULTI_CERTIFICATE_PACKAGE_LINE"
        elif len({row["dex_fingerprint"] for row in members}) == 1:
            classification = "SAME_DEX_CONTENT_VARIANTS"
        else:
            classification = "DIFFERENT_DEX_CONTENT_SAME_IDENTITY"
        group_rows.append({
            "package_name": key[0],
            "certificate_sha256": key[1],
            "version_code": key[2],
            "apk_count": len(members),
            "distinct_apk_sha256_count": len({row["sha256"] for row in members}),
            "distinct_dex_fingerprint_count": len({row["dex_fingerprint"] for row in members}),
            "distinct_manifest_fingerprint_count": len({row["manifest_fingerprint"] for row in members}),
            "years": "|".join(sorted({row["metadata_year"] for row in members})),
            "labels": "|".join(sorted({row["metadata_label"] for row in members})),
            "classification": classification,
        })
    write_csv(
        output / "collision_group_summary.csv",
        group_rows,
        [
            "package_name", "certificate_sha256", "version_code", "apk_count",
            "distinct_apk_sha256_count", "distinct_dex_fingerprint_count",
            "distinct_manifest_fingerprint_count", "years", "labels", "classification",
        ],
    )

    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P0_IDENTITY_COLLISION_FINGERPRINT",
        "formal_training_enabled": False,
        "lineage_csv": str(args.lineage.expanduser().resolve()),
        "selected_apk_count": len(fingerprints),
        "collision_group_count": sum(len(members) > 1 for members in collision_groups.values()),
        "multi_certificate_package_count": sum(len(certs) > 1 for certs in package_certs.values()),
        "fingerprint_status_counts": dict(Counter(row["fingerprint_status"] for row in fingerprints)),
        "group_classification_counts": dict(
            Counter(
                row["classification"]
                for row in group_rows
                if row["classification"] != "MULTI_CERTIFICATE_PACKAGE_LINE"
            )
        ),
        "multi_certificate_line_count": sum(
            row["classification"] == "MULTI_CERTIFICATE_PACKAGE_LINE"
            for row in group_rows
        ),
        "blocking_reasons": [
            "near_duplicate_clusters_not_built_for_full_inventory",
            "first_seen_utc_not_joined",
            "resolved_formal_training_config_not_generated",
        ],
        "notes": [
            "collision groups are not silently deduplicated",
            "DEX fingerprints are an initial triage signal, not a substitute for full near-duplicate clustering",
            "no call graph is built or persisted",
        ],
    }
    (output / "collision_fingerprint_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
