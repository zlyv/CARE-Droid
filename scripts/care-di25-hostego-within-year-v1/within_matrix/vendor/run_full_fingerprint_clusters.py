#!/usr/bin/env python3
"""Build full-inventory exact DEX and identity blocking clusters.

This is a conservative first pass. It does not claim fuzzy near-duplicate
clustering and never builds or persists call graphs.
"""

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
    result = {
        "sha256": row["sha256"],
        "absolute_path": row["absolute_path"],
        "package_name": row["package_name"],
        "certificate_sha256": row["certificate_sha256"],
        "version_code": row["version_code"],
        "version_name": row["version_name"],
        "dex_fingerprint": "",
        "manifest_fingerprint": "",
        "dex_entry_count": "",
        "zip_entry_count": "",
        "fingerprint_status": "ERROR",
        "error_type": "",
        "error_message": "",
    }
    try:
        with zipfile.ZipFile(row["absolute_path"], "r") as archive:
            names = archive.namelist()
            dex_names = [
                name for name in names
                if name.lower().startswith("classes") and name.lower().endswith(".dex")
            ]
            manifest_names = [
                name for name in names if name.lower() == "androidmanifest.xml"
            ]
            result["dex_entry_count"] = len(dex_names)
            result["zip_entry_count"] = len(names)
            result["dex_fingerprint"] = digest_entries(archive, dex_names) if dex_names else "NO_DEX"
            result["manifest_fingerprint"] = digest_entries(archive, manifest_names) if manifest_names else "NO_MANIFEST"
            result["fingerprint_status"] = "OK"
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)[:500]
    return result


class UnionFind:
    def __init__(self, keys):
        self.parent = {key: key for key in keys}

    def find(self, key):
        parent = self.parent[key]
        if parent != key:
            self.parent[key] = self.find(parent)
        return self.parent[key]

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def main():
    parser = argparse.ArgumentParser(description="Build conservative full APK fingerprint clusters")
    parser.add_argument("--lineage", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()
    rows = read_csv(args.lineage)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    fingerprints = []
    for index, row in enumerate(rows, start=1):
        fingerprints.append(fingerprint(row))
        if index % max(1, args.progress_every) == 0 or index == len(rows):
            print(f"processed {index}/{len(rows)}", flush=True)

    fingerprint_fields = [
        "sha256", "absolute_path", "package_name", "certificate_sha256", "version_code",
        "version_name", "dex_fingerprint", "manifest_fingerprint", "dex_entry_count",
        "zip_entry_count", "fingerprint_status", "error_type", "error_message",
    ]
    write_csv(output / "full_fingerprints.csv", fingerprints, fingerprint_fields)

    dex_groups = defaultdict(list)
    identity_groups = defaultdict(list)
    for row in fingerprints:
        if row["fingerprint_status"] != "OK":
            continue
        dex_groups[row["dex_fingerprint"]].append(row["sha256"])
        identity_groups[
            (row["package_name"], row["certificate_sha256"], row["version_code"])
        ].append(row["sha256"])

    keys = [row["sha256"] for row in fingerprints]
    union_find = UnionFind(keys)
    for members in dex_groups.values():
        for member in members[1:]:
            union_find.union(members[0], member)
    for members in identity_groups.values():
        if len(members) > 1:
            for member in members[1:]:
                union_find.union(members[0], member)

    components = defaultdict(list)
    for row in fingerprints:
        components[union_find.find(row["sha256"])].append(row)
    component_ids = {
        root: f"BLK{index:06d}"
        for index, root in enumerate(sorted(components), start=1)
    }
    assignment_rows = []
    for root, members in sorted(components.items()):
        dex_values = {row["dex_fingerprint"] for row in members}
        identity_values = {
            (row["package_name"], row["certificate_sha256"], row["version_code"])
            for row in members
        }
        reason = []
        if len(dex_values) < len(members):
            reason.append("EXACT_DEX")
        if len(identity_values) < len(members):
            reason.append("PACKAGE_CERT_VERSION")
        for row in members:
            assignment_rows.append({
                "sha256": row["sha256"],
                "absolute_path": row["absolute_path"],
                "package_name": row["package_name"],
                "certificate_sha256": row["certificate_sha256"],
                "version_code": row["version_code"],
                "block_id": component_ids[root],
                "block_size": len(members),
                "block_reason": "|".join(reason) if reason else "SINGLETON",
            })
    write_csv(
        output / "blocking_assignments.csv",
        assignment_rows,
        [
            "sha256", "absolute_path", "package_name", "certificate_sha256",
            "version_code", "block_id", "block_size", "block_reason",
        ],
    )

    block_sizes = [len(members) for members in components.values()]
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P0_FULL_EXACT_DEX_AND_IDENTITY_BLOCKING",
        "formal_training_enabled": False,
        "lineage_csv": str(args.lineage.expanduser().resolve()),
        "apk_count": len(fingerprints),
        "fingerprint_status_counts": dict(Counter(row["fingerprint_status"] for row in fingerprints)),
        "unique_dex_fingerprint_count": len(dex_groups),
        "exact_dex_group_count": sum(len(members) > 1 for members in dex_groups.values()),
        "identity_group_count": len(identity_groups),
        "identity_collision_group_count": sum(len(members) > 1 for members in identity_groups.values()),
        "blocking_component_count": len(components),
        "blocking_component_size_max": max(block_sizes) if block_sizes else 0,
        "blocking_component_size_mean": sum(block_sizes) / len(block_sizes) if block_sizes else 0.0,
        "outputs": {
            "fingerprints": str(output / "full_fingerprints.csv"),
            "assignments": str(output / "blocking_assignments.csv"),
        },
        "blocking_reasons": [
            "fuzzy_near_duplicate_similarity_not_calibrated",
            "first_seen_utc_not_joined",
            "static_parse_and_api_graph_coverage_not_available",
            "resolved_formal_training_config_not_generated",
        ],
        "notes": [
            "exact DEX fingerprints and package/certificate/version groups are combined conservatively",
            "blocking components are split constraints, not claims that all members are semantically identical",
            "no call graph is built or persisted",
        ],
    }
    (output / "full_fingerprint_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
