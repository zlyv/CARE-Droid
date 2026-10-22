#!/usr/bin/env python3
"""Package source code and non-sensitive profiles for the masking experiment.

This collector intentionally excludes APKs, feature rows, graph caches, model
weights, predictions, hashes, and labels.  It includes Python source, project
dependency manifests, small configuration files, selected audit JSON files,
and aggregate profiles of the 2017 feature/lineage/split tables.

Run from the tasa-droid repository root:

    python3 scripts/collect_role_masking_code_bundle.py --root .

Upload ``role_masking_code_bundle.zip`` after it completes.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


MAX_INCLUDED_FILE_BYTES = 2 * 1024 * 1024
MAX_CODE_FILES = 3000

SOURCE_ROOTS = ("scripts", "src", "tests")
SOURCE_SUFFIXES = {".py"}
CONFIG_SUFFIXES = {".toml", ".yaml", ".yml", ".ini", ".cfg"}
ROOT_CONFIG_PATTERNS = (
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "requirements-dev.txt",
    "environment.yml",
    "environment.yaml",
    "README.md",
)

SELECTED_AUDIT_PATTERNS = (
    re.compile(r"artifacts/(formal-p0-lineage|p0-lineage|p54-unified-lineage)/.*audit.*\.json$"),
    re.compile(r"artifacts/p1-splits-2017-temporal-inner(?:-v2)?/.*audit.*\.json$"),
    re.compile(r"artifacts/p9-2017-rolling-splits/.*audit.*\.json$"),
    re.compile(r"artifacts/p1-features/feature_schema\.json$"),
    re.compile(r"artifacts/p25-endpoint-cache-full-cap256/.*audit.*\.json$"),
    re.compile(r"artifacts/classifier-torch-2017/.*audit.*\.json$"),
)

PROFILE_CANDIDATES = {
    "features": (
        "artifacts/p1-features/features.csv",
        "artifacts/formal-p1-features/features.csv",
    ),
    "lineage": (
        "artifacts/clean-local-context-2017/lineage_2017.csv",
        "artifacts/formal-p0-lineage/package_certificate_lineage.csv",
        "artifacts/p0-lineage/package_certificate_lineage.csv",
    ),
    "temporal_split_v2": (
        "artifacts/p1-splits-2017-temporal-inner-v2/split_manifest.csv",
    ),
    "temporal_split": (
        "artifacts/p1-splits-2017-temporal-inner/split_manifest.csv",
    ),
    "rolling_split": (
        "artifacts/p9-2017-rolling-splits/split_manifest.csv",
    ),
    "method_context": (
        "artifacts/clean-local-context-2017/method_role_context.csv.gz",
    ),
    "contracted_edges": (
        "artifacts/clean-local-context-2017/contracted_structure_edges.csv.gz",
    ),
}


def rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def open_csv(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


def first_existing(root: Path, candidates):
    for candidate in candidates:
        path = root / candidate
        if path.is_file():
            return path
    return None


def safe_year(value):
    text = str(value or "").strip()
    match = re.match(r"^(19|20)\d{2}", text)
    return match.group(0) if match else "unknown"


def profile_features(path: Path):
    years = Counter()
    statuses = Counter()
    source_ids = Counter()
    lineage_present = 0
    rows = 0
    with open_csv(path) as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        for row in reader:
            rows += 1
            years[str(row.get("metadata_year", "")).strip() or "missing"] += 1
            statuses[str(row.get("status", "")).strip() or "missing"] += 1
            source_ids[str(row.get("source_id", "")).strip() or "missing"] += 1
            if str(row.get("lineage_id", "")).strip():
                lineage_present += 1

    role_nodes = [name for name in fields if name.startswith("role_node__")]
    role_edges = [name for name in fields if name.startswith("role_edge__")]
    context_roles = [name for name in fields if name.startswith("context_role__")]
    return {
        "path": str(path),
        "row_count": rows,
        "column_count": len(fields),
        "columns": fields,
        "year_counts": dict(years),
        "status_counts": dict(statuses),
        "source_id_counts": dict(source_ids),
        "lineage_id_present_count": lineage_present,
        "lineage_id_present_ratio": lineage_present / rows if rows else 0.0,
        "role_node_count": len(role_nodes),
        "role_edge_count": len(role_edges),
        "context_role_count": len(context_roles),
        "role_node_names": role_nodes,
        "role_edge_names": role_edges,
    }


def profile_lineage(path: Path):
    group_sizes = Counter()
    years = Counter()
    dex_years = Counter()
    labels = Counter()
    row_count = 0
    missing_lineage = 0
    with open_csv(path) as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        for row in reader:
            row_count += 1
            lineage = str(row.get("lineage_id", "")).strip()
            if lineage:
                group_sizes[lineage] += 1
            else:
                missing_lineage += 1
            years[str(row.get("metadata_year", "")).strip() or "missing"] += 1
            dex_years[safe_year(row.get("dex_date"))] += 1
            labels[str(row.get("metadata_label", "")).strip() or "missing"] += 1

    size_histogram = Counter(group_sizes.values())
    multi_rows = sum(size for size in group_sizes.values() if size > 1)
    return {
        "path": str(path),
        "columns": fields,
        "row_count": row_count,
        "missing_lineage_count": missing_lineage,
        "unique_lineage_count": len(group_sizes),
        "singleton_lineage_count": int(size_histogram.get(1, 0)),
        "multi_member_lineage_count": sum(1 for size in group_sizes.values() if size > 1),
        "rows_in_multi_member_lineages": multi_rows,
        "rows_in_multi_member_lineages_ratio": multi_rows / row_count if row_count else 0.0,
        "largest_lineage_size": max(group_sizes.values()) if group_sizes else 0,
        "lineage_size_histogram": {str(k): v for k, v in sorted(size_histogram.items())},
        "metadata_year_counts": dict(years),
        "dex_year_counts": dict(dex_years),
        "metadata_label_counts": dict(labels),
    }


def profile_split(path: Path):
    split_partition = Counter()
    split_rows = Counter()
    exclusions = Counter()
    blocking_groups = defaultdict(set)
    labels = Counter()
    row_count = 0
    with open_csv(path) as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        for row in reader:
            row_count += 1
            split_name = str(row.get("split_name", "")).strip() or "missing"
            partition = str(row.get("partition", "")).strip() or "missing"
            split_partition[(split_name, partition)] += 1
            split_rows[split_name] += 1
            exclusions[str(row.get("exclusion_reason", "")).strip() or "none"] += 1
            group = str(row.get("blocking_group", "")).strip()
            if group:
                blocking_groups[split_name].add(group)
            labels[str(row.get("label", "")).strip() or "missing"] += 1
    return {
        "path": str(path),
        "columns": fields,
        "row_count": row_count,
        "split_row_counts": dict(split_rows),
        "split_partition_counts": {
            f"{split_name}::{partition}": count
            for (split_name, partition), count in sorted(split_partition.items())
        },
        "blocking_group_counts": {
            key: len(value) for key, value in blocking_groups.items()
        },
        "exclusion_reason_counts": dict(exclusions),
        "label_counts": dict(labels),
    }


def profile_large_context_table(path: Path):
    row_count = 0
    apk_ids = set()
    target_types = Counter()
    role_strings_present = 0
    with open_csv(path) as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        for row in reader:
            row_count += 1
            sha = str(row.get("sha256", "")).strip()
            if sha:
                apk_ids.add(sha)
            if "target_type" in row:
                target_types[str(row.get("target_type", "")).strip() or "missing"] += 1
            if str(row.get("anchor_roles", "")).strip() or str(row.get("context_role", "")).strip():
                role_strings_present += 1
    return {
        "path": str(path),
        "columns": fields,
        "row_count": row_count,
        "unique_apk_count": len(apk_ids),
        "target_type_counts": dict(target_types),
        "rows_with_role_strings": role_strings_present,
    }


def collect_profiles(root: Path):
    profiles = {}
    for name, candidates in PROFILE_CANDIDATES.items():
        path = first_existing(root, candidates)
        if path is None:
            profiles[name] = {"found": False, "candidates": list(candidates)}
            continue
        try:
            if name == "features":
                value = profile_features(path)
            elif name == "lineage":
                value = profile_lineage(path)
            elif "split" in name:
                value = profile_split(path)
            else:
                value = profile_large_context_table(path)
            value["found"] = True
            value["path"] = rel(path, root)
            profiles[name] = value
        except Exception as exc:
            profiles[name] = {
                "found": True,
                "path": rel(path, root),
                "error": str(exc),
            }
    return profiles


def source_files(root: Path):
    selected = []
    for root_name in SOURCE_ROOTS:
        base = root / root_name
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
                continue
            if ".bak" in path.name or "__pycache__" in path.parts:
                continue
            if path.stat().st_size <= MAX_INCLUDED_FILE_BYTES:
                selected.append(path)
            if len(selected) >= MAX_CODE_FILES:
                raise RuntimeError(f"Source file cap exceeded ({MAX_CODE_FILES})")

    for name in ROOT_CONFIG_PATTERNS:
        path = root / name
        if path.is_file() and path.stat().st_size <= MAX_INCLUDED_FILE_BYTES:
            selected.append(path)

    for directory_name in ("config", "configs", "resources"):
        base = root / directory_name
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower() in CONFIG_SUFFIXES | {".json"}
                and path.stat().st_size <= MAX_INCLUDED_FILE_BYTES
            ):
                selected.append(path)

    for path in (root / "artifacts").rglob("*.json") if (root / "artifacts").is_dir() else ():
        relative = rel(path, root)
        if any(pattern.search(relative) for pattern in SELECTED_AUDIT_PATTERNS):
            if path.stat().st_size <= MAX_INCLUDED_FILE_BYTES:
                selected.append(path)

    unique = {}
    for path in selected:
        unique[rel(path, root)] = path
    return [unique[key] for key in sorted(unique)]


def render_report(profile, included):
    lines = [
        "ROLE MASKING CODE BUNDLE",
        "=" * 80,
        f"Python: {sys.version.replace(os.linesep, ' ')}",
        f"Included source/config/audit files: {len(included)}",
        "",
        "DATA PROFILES",
        "-" * 80,
    ]
    for name, value in profile.items():
        lines.append(name)
        lines.append(json.dumps(value, ensure_ascii=False, indent=2))
    lines.extend([
        "",
        "The archive deliberately excludes APKs, feature rows, graph cache rows,",
        "predictions, SHA lists, labels, and model weights.",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Collect role-masking implementation context.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Default: <root>/role_masking_code_bundle.zip",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Repository root is not a directory: {root}")
    archive_path = (args.archive or (root / "role_masking_code_bundle.zip")).resolve()

    profiles = collect_profiles(root)
    included = source_files(root)
    profile_json = json.dumps(profiles, ensure_ascii=False, indent=2).encode("utf-8")
    report = render_report(profiles, included).encode("utf-8")

    if archive_path.exists():
        archive_path.unlink()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("role_masking_code_bundle/data_profiles.json", profile_json)
        archive.writestr("role_masking_code_bundle/REPORT.txt", report)
        for path in included:
            archive.write(path, arcname="role_masking_code_bundle/repository/" + rel(path, root))

    print(f"Included files: {len(included)}")
    print(f"Upload this archive: {archive_path}")


if __name__ == "__main__":
    main()
