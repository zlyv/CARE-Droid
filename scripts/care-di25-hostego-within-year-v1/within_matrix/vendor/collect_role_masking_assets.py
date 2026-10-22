#!/usr/bin/env python3
"""Collect a compact, non-destructive capability audit for role masking.

Run this script from the repository root.  It does not copy APKs, feature
rows, hashes, labels, or graph contents.  It records schemas, file counts,
dependency availability, and a bounded set of candidate paths/code references.

Example
-------
python scripts/collect_role_masking_assets.py --root .

The command creates ``role_masking_asset_audit.zip`` in the repository root.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import importlib.util
import json
import os
import platform
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


AUDIT_VERSION = 1
DEFAULT_MAX_FILES = 300000
MAX_CANDIDATES_PER_GROUP = 200
MAX_CODE_HITS = 500
MAX_SCHEMA_DEPTH = 5
MAX_SCHEMA_ITEMS = 30

EXCLUDED_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    "__pycache__",
    "node_modules",
    "site-packages",
    ".mypy_cache",
    ".pytest_cache",
}

GRAPH_SUFFIXES = {
    ".gexf",
    ".gml",
    ".graphml",
    ".gpickle",
    ".dot",
    ".gv",
    ".edgelist",
    ".adjlist",
}

STRUCTURED_SUFFIXES = {
    ".csv",
    ".gz",
    ".json",
    ".jsonl",
    ".npz",
    ".npy",
    ".parquet",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
}

CANDIDATE_PATTERNS = {
    "api": re.compile(r"api|method|invoke|call", re.I),
    "graph": re.compile(r"graph|edge|node|caller|callee|endpoint", re.I),
    "role": re.compile(r"role|semantic|behavior|behaviour", re.I),
    "family": re.compile(r"family|avclass|cluster|lineage", re.I),
    "time": re.compile(r"first.?seen|timestamp|release|date|year|month", re.I),
    "experiment": re.compile(r"split|manifest|feature|prediction|cache", re.I),
}

CODE_PATTERN = re.compile(
    r"ROLE_|role_|api_|API_|call.?graph|caller|callee|family|avclass|"
    r"first.?seen|timestamp|mask|contrast|augment",
    re.I,
)

IMPORTANT_EXACT_NAMES = {
    "features.csv",
    "feature_schema.json",
    "endpoint_centered_tensor_cache.jsonl.gz",
    "p72_oof_predictions.csv",
    "split_manifest.csv",
    "run_allratio70_vs_nodrop76_experiment.py",
}


def relative_text(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return str(path.resolve())


def bounded_append(container, value, limit=MAX_CANDIDATES_PER_GROUP):
    if len(container) < limit:
        container.append(value)


def summarize_value(value, depth=0):
    """Return structure only; deliberately omit scalar values."""
    if depth >= MAX_SCHEMA_DEPTH:
        return {"type": type(value).__name__, "truncated": True}
    if isinstance(value, dict):
        keys = list(value.keys())
        shown = keys[:MAX_SCHEMA_ITEMS]
        return {
            "type": "dict",
            "key_count": len(keys),
            "keys": [str(key) for key in shown],
            "children": {
                str(key): summarize_value(value[key], depth + 1)
                for key in shown
            },
            "truncated": len(keys) > len(shown),
        }
    if isinstance(value, (list, tuple)):
        sample = list(value[: min(3, len(value))])
        return {
            "type": type(value).__name__,
            "length": len(value),
            "item_schemas": [summarize_value(item, depth + 1) for item in sample],
        }
    if value is None:
        return {"type": "null"}
    if isinstance(value, str):
        return {"type": "str", "length": len(value)}
    return {"type": type(value).__name__}


def read_csv_header(path: Path):
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    mode = "rt"
    with opener(path, mode, encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def inspect_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return summarize_value(json.load(handle))


def inspect_feature_schema(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    fields = value.get("feature_fields", []) if isinstance(value, dict) else []
    return {
        "structure": summarize_value(value),
        "feature_fields": [str(field) for field in fields],
        "graph_persistence": value.get("graph_persistence") if isinstance(value, dict) else None,
    }


def inspect_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                return summarize_value(json.loads(line))
    return {"type": "empty"}


def package_status():
    names = (
        "numpy",
        "scipy",
        "sklearn",
        "torch",
        "torch_geometric",
        "networkx",
        "pandas",
    )
    result = {}
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover - Python 3.7 compatibility
        metadata = None

    distribution_names = {
        "sklearn": "scikit-learn",
        "torch_geometric": "torch-geometric",
    }
    for name in names:
        available = importlib.util.find_spec(name) is not None
        version = None
        if available and metadata is not None:
            try:
                version = metadata.version(distribution_names.get(name, name))
            except Exception:
                version = None
        result[name] = {"available": available, "version": version}
    return result


def local_imports(path: Path):
    try:
        source = path.read_text(encoding="utf-8-sig", errors="replace")
        tree = ast.parse(source, filename=str(path))
    except Exception as exc:
        return {"error": str(exc)}

    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module.split(".")[0])

    standard = set(getattr(sys, "stdlib_module_names", ()))
    result = []
    for name in sorted(set(names)):
        if name in standard:
            continue
        sibling = path.parent / (name + ".py")
        result.append({
            "module": name,
            "sibling_exists": sibling.is_file(),
            "importable_from_environment": importlib.util.find_spec(name) is not None,
        })
    return result


def scan_repository(root: Path, max_files: int):
    candidates = defaultdict(list)
    exact = []
    suffix_counts = Counter()
    graph_suffix_counts = Counter()
    python_files = []
    structured_files = []
    total_files = 0
    truncated = False

    for current, dir_names, file_names in os.walk(root):
        dir_names[:] = [
            name for name in dir_names
            if name not in EXCLUDED_DIR_NAMES
        ]
        current_path = Path(current)
        for name in file_names:
            total_files += 1
            if total_files > max_files:
                truncated = True
                break
            path = current_path / name
            rel = relative_text(path, root)
            suffix = path.suffix.lower()
            suffix_counts[suffix or "<none>"] += 1
            if suffix in GRAPH_SUFFIXES:
                graph_suffix_counts[suffix] += 1
            if suffix == ".py":
                python_files.append(path)
            if suffix in STRUCTURED_SUFFIXES:
                structured_files.append(path)
            if name in IMPORTANT_EXACT_NAMES:
                bounded_append(exact, rel, limit=500)
            searchable = rel.replace("\\", "/")
            for group, pattern in CANDIDATE_PATTERNS.items():
                if pattern.search(searchable):
                    bounded_append(candidates[group], rel)
        if truncated:
            break

    return {
        "total_files_scanned": min(total_files, max_files),
        "scan_truncated": truncated,
        "suffix_counts": dict(suffix_counts.most_common(80)),
        "graph_suffix_counts": dict(graph_suffix_counts),
        "exact_candidates": exact,
        "candidates": dict(candidates),
    }, python_files, structured_files


def inspect_structured_files(paths, root):
    result = []
    for path in paths:
        lower = path.name.lower()
        should_inspect = (
            path.name in IMPORTANT_EXACT_NAMES
            or any(pattern.search(lower) for pattern in CANDIDATE_PATTERNS.values())
        )
        if not should_inspect:
            continue
        row = {
            "path": relative_text(path, root),
            "bytes": path.stat().st_size,
        }
        try:
            if lower.endswith(".csv") or lower.endswith(".csv.gz"):
                header = read_csv_header(path)
                row["kind"] = "csv"
                row["column_count"] = len(header)
                row["columns"] = header
            elif lower.endswith("feature_schema.json"):
                row["kind"] = "json_schema"
                row.update(inspect_feature_schema(path))
            elif lower.endswith("endpoint_centered_tensor_cache.jsonl.gz"):
                row["kind"] = "jsonl_gz_first_record_schema"
                row["schema"] = inspect_jsonl_gz(path)
            else:
                continue
        except Exception as exc:
            row["inspection_error"] = str(exc)
        result.append(row)
        if len(result) >= 300:
            break
    return result


def code_hits(python_files, root):
    hits = []
    for path in python_files:
        try:
            with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
                for line_number, line in enumerate(handle, 1):
                    if CODE_PATTERN.search(line):
                        bounded_append(
                            hits,
                            {
                                "path": relative_text(path, root),
                                "line": line_number,
                                "text": line.strip()[:500],
                            },
                            limit=MAX_CODE_HITS,
                        )
                    if len(hits) >= MAX_CODE_HITS:
                        return hits
        except OSError:
            continue
    return hits


def import_audits(python_files, root):
    result = []
    for path in python_files:
        if path.name in {
            "run_allratio70_vs_nodrop76_experiment.py",
            "run_semantic_collision_frozen_future_eval.py",
        }:
            result.append({
                "path": relative_text(path, root),
                "imports": local_imports(path),
            })
    return result


def capability_assessment(audit):
    structured = audit["structured_file_schemas"]
    columns = set()
    cache_schema_found = False
    for item in structured:
        columns.update(item.get("columns", []))
        columns.update(item.get("feature_fields", []))
        if item.get("kind") == "jsonl_gz_first_record_schema":
            cache_schema_found = True

    lower_columns = {str(value).casefold() for value in columns}
    api_identity_markers = (
        "api_calls",
        "api_sequence",
        "api_name",
        "api_names",
        "invocations",
        "invoke_targets",
        "caller",
        "callee",
    )
    has_api_identity_columns = any(
        marker in value
        for value in lower_columns
        for marker in api_identity_markers
    )
    has_family_columns = any(
        "family" in value or "avclass" in value or "lineage" in value
        for value in lower_columns
    )
    has_fine_time = any(
        key in value
        for value in lower_columns
        for key in ("month", "timestamp", "first_seen", "firstseen", "date")
    )
    graph_count = sum(audit["repository_scan"]["graph_suffix_counts"].values())

    return {
        "role_feature_level_masking": {
            "supported": any(
                value.startswith("role_node__") or value.startswith("role_edge__")
                for value in columns
            ),
            "reason": "Requires role_node__/role_edge__ feature columns.",
        },
        "endpoint_structure_expert": {
            "supported": cache_schema_found,
            "reason": "Requires endpoint_centered_tensor_cache.jsonl.gz.",
        },
        "api_level_masking_or_substitution": {
            "supported_by_detected_metadata": bool(has_api_identity_columns or graph_count),
            "reason": "Needs per-APK API identities or raw graph nodes, not only aggregates.",
        },
        "family_holdout_pseudo_drift": {
            "supported_by_detected_metadata": has_family_columns,
            "reason": "Needs family/AVClass labels or a separately defined source-only clustering step.",
        },
        "within_2017_temporal_split": {
            "supported_by_detected_metadata": has_fine_time,
            "reason": "Needs month/date/first-seen rather than metadata_year alone.",
        },
    }


def render_report(audit):
    lines = []
    lines.append("ROLE MASKING ASSET AUDIT")
    lines.append("=" * 80)
    lines.append(f"Audit version: {audit['audit_version']}")
    lines.append(f"Root: {audit['root']}")
    lines.append(f"Python: {audit['environment']['python']}")
    lines.append(f"Files scanned: {audit['repository_scan']['total_files_scanned']}")
    lines.append(f"Scan truncated: {audit['repository_scan']['scan_truncated']}")
    lines.append("")
    lines.append("CAPABILITY ASSESSMENT")
    lines.append("-" * 80)
    for name, status in audit["capability_assessment"].items():
        supported = status.get("supported", status.get("supported_by_detected_metadata"))
        lines.append(f"{name}: {supported}")
        lines.append(f"  {status['reason']}")
    lines.append("")
    lines.append("PACKAGE AVAILABILITY")
    lines.append("-" * 80)
    for name, status in audit["environment"]["packages"].items():
        lines.append(f"{name}: {status['available']} version={status['version']}")
    lines.append("")
    lines.append("EXACT ASSET CANDIDATES")
    lines.append("-" * 80)
    for path in audit["repository_scan"]["exact_candidates"]:
        lines.append(path)
    lines.append("")
    lines.append("STRUCTURED FILE SCHEMAS")
    lines.append("-" * 80)
    for item in audit["structured_file_schemas"]:
        lines.append(f"{item['path']} ({item.get('kind', 'unknown')})")
        if "columns" in item:
            lines.append("  columns: " + ", ".join(item["columns"]))
        if "schema" in item:
            lines.append("  schema: " + json.dumps(item["schema"], ensure_ascii=False))
        if "feature_fields" in item:
            lines.append("  feature_fields: " + ", ".join(item["feature_fields"]))
        if "structure" in item:
            lines.append("  structure: " + json.dumps(item["structure"], ensure_ascii=False))
        if "graph_persistence" in item:
            lines.append("  graph_persistence: " + str(item["graph_persistence"]))
        if "inspection_error" in item:
            lines.append("  error: " + item["inspection_error"])
    lines.append("")
    lines.append("See audit.json for bounded candidate paths, code references, and imports.")
    return "\n".join(lines) + "\n"


def write_outputs(root: Path, output_dir: Path, archive_path: Path, audit):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "audit.json"
    report_path = output_dir / "REPORT.txt"
    json_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path.write_text(render_report(audit), encoding="utf-8")

    if archive_path.exists():
        archive_path.unlink()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(json_path, arcname="role_masking_asset_audit/audit.json")
        archive.write(report_path, arcname="role_masking_asset_audit/REPORT.txt")

    return json_path, report_path, archive_path


def main():
    parser = argparse.ArgumentParser(
        description="Audit repository assets needed for role-aware masking experiments."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <root>/role_masking_asset_audit",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Default: <root>/role_masking_asset_audit.zip",
    )
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Repository root is not a directory: {root}")
    if args.max_files < 1:
        raise SystemExit("--max-files must be positive")

    output_dir = (args.output_dir or (root / "role_masking_asset_audit")).resolve()
    archive_path = (args.archive or (root / "role_masking_asset_audit.zip")).resolve()

    scan, python_files, structured_files = scan_repository(root, args.max_files)
    audit = {
        "audit_version": AUDIT_VERSION,
        "root": str(root),
        "environment": {
            "python": sys.version.replace("\n", " "),
            "executable": sys.executable,
            "platform": platform.platform(),
            "packages": package_status(),
        },
        "repository_scan": scan,
        "structured_file_schemas": inspect_structured_files(structured_files, root),
        "important_import_audits": import_audits(python_files, root),
        "bounded_code_hits": code_hits(python_files, root),
    }
    audit["capability_assessment"] = capability_assessment(audit)

    json_path, report_path, archive_path = write_outputs(
        root,
        output_dir,
        archive_path,
        audit,
    )
    print(f"Audit JSON: {json_path}")
    print(f"Short report: {report_path}")
    print(f"Upload this archive: {archive_path}")


if __name__ == "__main__":
    main()
