#!/usr/bin/env python3
"""Rebuild the exact future lineage with the 1,989 frozen shards as authority."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_BY_YEAR = {2018: 793, 2019: 399, 2020: 397, 2021: 400}


def norm(value) -> str:
    return str(value or "").strip().lower()


def first(row: dict, names: tuple[str, ...]) -> str:
    for name in names:
        value = str(row.get(name, "")).strip()
        if value:
            return value
    return ""


def label(value) -> str:
    value = norm(value)
    if value in {"0", "0.0", "benign", "goodware", "good"}: return "benign"
    if value in {"1", "1.0", "malware", "malicious", "badware", "bad"}: return "malware"
    return ""


def remap(value: str, storage: Path) -> Path:
    text = str(value or "").strip()
    for prefix in ("/home/boyuanjie/datasets/", "/home/boyuanjie/projects/tasa-droid/datasets/"):
        if text.startswith(prefix): return storage / "datasets" / text[len(prefix):]
    path = Path(text)
    return path if path.is_absolute() else storage / "datasets" / path


def read(path: Path) -> tuple[list[str], list[dict]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle); return list(reader.fieldnames or []), list(reader)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    shard_paths = sorted(args.shards.glob("*.npz"))
    shard_ids = [path.stem.lower() for path in shard_paths]
    if len(shard_ids) != len(set(shard_ids)) or len(shard_ids) != 1989 or any(path.stat().st_size == 0 for path in shard_paths):
        raise RuntimeError("BLOCKED_BY_CONTRACT: frozen shard cohort is not exactly 1,989 unique non-empty NPZ files")
    authority = set(shard_ids)

    _, manifest_rows = read(args.manifest)
    grouped = defaultdict(list)
    for row in manifest_rows:
        sha = norm(first(row, ("sha256", "sample_id", "apk_id")))
        if sha in authority: grouped[sha].append(row)
    _, status_rows = read(args.status)
    status_by_sha = {}
    duplicate_status = []
    for row in status_rows:
        sha = norm(first(row, ("sha256", "sample_id", "apk_id")))
        if sha in status_by_sha: duplicate_status.append(sha)
        elif sha: status_by_sha[sha] = row

    output_rows = []
    conflicts = []
    missing_manifest = []
    missing_status = []
    bad_status = []
    missing_apks = []
    for sha in shard_ids:
        candidates = grouped.get(sha, [])
        if not candidates:
            missing_manifest.append(sha); continue
        signatures = {
            (first(row, ("absolute_path", "apk_path", "path", "file_path")),
             first(row, ("metadata_year", "year")),
             label(first(row, ("metadata_label", "label"))))
            for row in candidates
        }
        if len(signatures) != 1:
            conflicts.append(sha); continue
        row = candidates[0]
        status = status_by_sha.get(sha)
        if status is None:
            missing_status.append(sha); continue
        if norm(status.get("status")) not in {"ok", "success"}:
            bad_status.append(sha); continue
        path = remap(first(row, ("absolute_path", "apk_path", "path", "file_path")), args.storage_root)
        if not path.is_file(): missing_apks.append(sha)
        year = int(float(first(row, ("metadata_year", "year"))))
        value = label(first(row, ("metadata_label", "label")))
        output_rows.append({
            "sha256": sha, "absolute_path": str(path.resolve()),
            "source_id": first(row, ("source_id",)) or "future_2018_2021",
            "lineage_id": first(row, ("lineage_id",)) or sha,
            "metadata_year": str(year), "metadata_label": value,
        })
    output_rows.sort(key=lambda row: (int(row["metadata_year"]), row["sha256"]))
    output_ids = [row["sha256"] for row in output_rows]
    years = Counter(int(row["metadata_year"]) for row in output_rows)
    bad_labels = [row["sha256"] for row in output_rows if row["metadata_label"] not in {"benign", "malware"}]
    checks = {
        "shards_exact_unique_nonempty_1989": len(authority) == 1989,
        "output_exact_unique_1989": len(output_ids) == len(set(output_ids)) == 1989,
        "output_set_equals_shards": set(output_ids) == authority,
        "year_counts_exact": dict(years) == EXPECTED_BY_YEAR,
        "manifest_missing_zero": not missing_manifest,
        "manifest_conflicts_zero": not conflicts,
        "status_missing_zero": not missing_status,
        "status_bad_zero": not bad_status,
        "status_ids_unique": not duplicate_status,
        "apk_paths_exist": not missing_apks,
        "labels_binary": not bad_labels,
    }
    passed = all(checks.values())
    audit = {
        "passed": passed, "checks": checks, "year_counts": dict(years),
        "missing_manifest": missing_manifest, "manifest_conflicts": conflicts,
        "missing_status": missing_status, "bad_status": bad_status,
        "duplicate_status": duplicate_status, "missing_apks": missing_apks,
        "bad_labels": bad_labels, "cohort_authority": str(args.shards.resolve()),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("BLOCKED_BY_CONTRACT: future lineage reconstruction failed; see " + str(args.audit))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0])); writer.writeheader(); writer.writerows(output_rows)
    print("FUTURE_LINEAGE_REBUILT_FROM_1989_FROZEN_SHARDS")
    print("year_counts =", dict(years))


if __name__ == "__main__":
    main()
