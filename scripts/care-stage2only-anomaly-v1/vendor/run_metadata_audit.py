#!/usr/bin/env python3
"""Join enabled-source CSV metadata to the P0 inventory by SHA-256.

This script intentionally keeps dex_date separate from first_seen_utc.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


LABELS = {
    "0": "benign",
    "benign": "benign",
    "clean": "benign",
    "negative": "benign",
    "1": "malicious",
    "malware": "malicious",
    "malicious": "malicious",
    "positive": "malicious",
    "virus": "malicious",
}


def norm_hash(value):
    return str(value or "").strip().lower()


def norm_label(value):
    return LABELS.get(str(value or "").strip().lower())


def as_int(value):
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metadata(config):
    dataset_root = Path(config["dataset_root"]).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (Path.cwd() / dataset_root).resolve()

    records = defaultdict(list)
    file_reports = []
    seen_files = set()

    for source in config.get("sources", []):
        if not source.get("enabled", True):
            continue
        source_id = str(source.get("id", ""))
        for configured_root in source.get("list_roots", []):
            root = Path(str(configured_root)).expanduser()
            if not root.is_absolute():
                root = dataset_root / root
            root = root.resolve()
            if not root.is_dir():
                file_reports.append({
                    "source_id": source_id,
                    "root": str(root),
                    "status": "MISSING_ROOT",
                    "row_count": 0,
                })
                continue

            for path in sorted(root.rglob("*.csv")):
                path = path.resolve()
                if path in seen_files:
                    continue
                seen_files.add(path)
                row_count = 0
                with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
                    reader = csv.DictReader(handle)
                    if "sha256" not in (reader.fieldnames or []):
                        file_reports.append({
                            "source_id": source_id,
                            "file": str(path),
                            "status": "SKIPPED_MISSING_SHA256_COLUMN",
                            "row_count": 0,
                        })
                        continue
                    for row in reader:
                        digest = norm_hash(row.get("sha256"))
                        if len(digest) != 64:
                            continue
                        row_count += 1
                        records[digest].append({
                            "source_id": source_id,
                            "metadata_file": str(path),
                            "metadata_year": as_int(row.get("year")),
                            "metadata_label": norm_label(row.get("label")),
                            "metadata_label_name": str(row.get("label_name") or "").strip(),
                            "metadata_apk_size": as_int(row.get("apk_size")),
                            "dex_date": str(row.get("dex_date") or "").strip(),
                            "vt_scan_date": str(row.get("vt_scan_date") or "").strip(),
                            "markets": str(row.get("markets") or "").strip(),
                            "metadata_pkg_name": str(row.get("pkg_name") or "").strip(),
                        })
                file_reports.append({
                    "source_id": source_id,
                    "file": str(path),
                    "status": "READ",
                    "row_count": row_count,
                })
    return records, file_reports


def conflict_fields(rows):
    fields = (
        "metadata_year",
        "metadata_label",
        "metadata_label_name",
        "metadata_apk_size",
        "dex_date",
        "metadata_pkg_name",
    )
    return [field for field in fields if len({row.get(field) for row in rows}) > 1]


def main():
    parser = argparse.ArgumentParser(description="Audit old APK metadata joins")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve(strict=True)
    inventory_path = args.inventory.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    metadata, file_reports = load_metadata(config)
    with inventory_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        inventory = list(csv.DictReader(handle))

    fields = [
        "source_id", "sha256", "absolute_path", "inferred_year", "inferred_label",
        "metadata_year", "metadata_label", "metadata_label_name", "metadata_apk_size",
        "dex_date", "vt_scan_date", "markets", "metadata_pkg_name", "metadata_file",
        "join_status", "year_consistent", "label_consistent", "size_consistent",
        "first_seen_utc",
    ]
    joined = []
    statuses = Counter()
    mismatches = Counter()
    inventory_hashes = set()

    for item in inventory:
        digest = norm_hash(item.get("sha256"))
        inventory_hashes.add(digest)
        candidates = metadata.get(digest, [])
        conflicts = conflict_fields(candidates) if candidates else []
        row = candidates[0] if candidates else {}
        status = "MISSING_METADATA" if not candidates else (
            "CONFLICTING_METADATA" if conflicts else "MATCHED"
        )
        inferred_year = as_int(item.get("year"))
        inferred_label = norm_label(item.get("label"))
        metadata_year = row.get("metadata_year")
        metadata_label = row.get("metadata_label")
        metadata_size = row.get("metadata_apk_size")
        year_ok = None if metadata_year is None else inferred_year == metadata_year
        label_ok = None if metadata_label is None else inferred_label == metadata_label
        size_ok = None if metadata_size is None else as_int(item.get("size_bytes")) == metadata_size
        if year_ok is False:
            mismatches["year_mismatch"] += 1
        if label_ok is False:
            mismatches["label_mismatch"] += 1
        if size_ok is False:
            mismatches["size_mismatch"] += 1
        statuses[status] += 1
        joined.append({
            "source_id": item.get("source_id", ""),
            "sha256": digest,
            "absolute_path": item.get("absolute_path", ""),
            "inferred_year": inferred_year or "",
            "inferred_label": inferred_label or "",
            "metadata_year": metadata_year if row else "",
            "metadata_label": metadata_label or "",
            "metadata_label_name": row.get("metadata_label_name", ""),
            "metadata_apk_size": metadata_size if row else "",
            "dex_date": row.get("dex_date", ""),
            "vt_scan_date": row.get("vt_scan_date", ""),
            "markets": row.get("markets", ""),
            "metadata_pkg_name": row.get("metadata_pkg_name", ""),
            "metadata_file": row.get("metadata_file", ""),
            "join_status": status,
            "year_consistent": year_ok if year_ok is not None else "",
            "label_consistent": label_ok if label_ok is not None else "",
            "size_consistent": size_ok if size_ok is not None else "",
            "first_seen_utc": "",
        })

    write_csv(output / "metadata_join.csv", joined, fields)
    unmatched = [
        dict(row, sha256=digest)
        for digest, rows in metadata.items()
        if digest not in inventory_hashes
        for row in rows
    ]
    write_csv(output / "metadata_not_in_inventory.csv", unmatched, [
        "source_id", "sha256", "metadata_file", "metadata_year", "metadata_label",
        "metadata_label_name", "metadata_apk_size", "dex_date", "vt_scan_date",
        "markets", "metadata_pkg_name",
    ])

    conflict_rows = []
    for digest, rows in sorted(metadata.items()):
        fields_with_conflicts = conflict_fields(rows)
        if fields_with_conflicts:
            conflict_rows.append({
                "sha256": digest,
                "metadata_row_count": len(rows),
                "conflict_fields": json.dumps(fields_with_conflicts),
                "metadata_files": json.dumps(sorted({r["metadata_file"] for r in rows})),
            })
    write_csv(output / "metadata_conflicts.csv", conflict_rows, [
        "sha256", "metadata_row_count", "conflict_fields", "metadata_files",
    ])

    matched = statuses["MATCHED"]
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P0_METADATA_JOIN",
        "p0_status": "METADATA_JOINED_NOT_AUDITED",
        "formal_training_enabled": False,
        "config_path": str(config_path),
        "inventory_csv": str(inventory_path),
        "inventory_sha256": sha256_file(inventory_path),
        "inventory_apk_count": len(inventory),
        "metadata_file_count": len(file_reports),
        "metadata_row_count": sum(r.get("row_count", 0) for r in file_reports),
        "metadata_unique_sha256_count": len(metadata),
        "join_status_counts": dict(sorted(statuses.items())),
        "joined_count": matched,
        "join_coverage": matched / len(inventory) if inventory else 0.0,
        "mismatch_counts": dict(sorted(mismatches.items())),
        "metadata_conflict_group_count": len(conflict_rows),
        "first_seen_utc_joined_count": 0,
        "certificate_sha256_joined_count": 0,
        "file_reports": file_reports,
        "blocking_reasons": [
            "audited_first_seen_utc_not_joined",
            "package_and_certificate_version_lines_not_built",
            "near_duplicate_clusters_not_built",
            "static_parse_and_api_graph_coverage_not_available",
            "resolved_formal_training_config_not_generated",
        ],
        "notes": [
            "join key is SHA256",
            "dex_date is not first_seen_utc",
            "disabled sources are excluded",
        ],
    }
    (output / "metadata_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
