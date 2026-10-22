#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from common import OUTPUT, PROJECT, dump_json, ensure_output, normalize_owner, read_csv, sha256, truthy


NAME_TOKENS = (
    "libscout", "libradar", "libid", "libscan", "libdetect", "tpl", "thirdparty",
    "third-party", "sdk-profile", "sdk_profile", "library-profile", "library_profile",
)
IGNORE_DIRS = {".git", "__pycache__", "node_modules", "cache", "checkpoints", "tmp"}


def candidate_roots() -> list[Path]:
    values = [
        PROJECT / "scripts", PROJECT / "tools", PROJECT / "vendor", PROJECT / "artifacts", PROJECT / "outputs",
        Path("/home/boyuanjie/tools"), Path("/home/boyuanjie/projects"),
        Path("/mnt/HDD1/boyuanjie/tasa-droid-storage"),
    ]
    extra = os.environ.get("PASEA_SEARCH_ROOTS", "")
    values.extend(Path(x).expanduser() for x in extra.split(os.pathsep) if x.strip())
    result, seen = [], set()
    for path in values:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved not in seen and resolved.exists():
            seen.add(resolved)
            result.append(resolved)
    return result


def scan(root: Path, max_depth: int = 5, cap: int = 2000) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    root_depth = len(root.parts)
    for base, dirs, files in os.walk(root):
        depth = len(Path(base).parts) - root_depth
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and depth < max_depth]
        names = [(Path(base) / d, "directory") for d in dirs] + [(Path(base) / f, "file") for f in files]
        for path, kind in names:
            lowered = path.name.lower()
            if not any(token in lowered for token in NAME_TOKENS):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            record = {"path": str(path.resolve()), "kind": kind, "size": stat.st_size}
            if kind == "file" and stat.st_size <= 64 * 1024 * 1024:
                try:
                    record["sha256"] = sha256(path)
                except OSError:
                    record["sha256"] = None
            records.append(record)
            if len(records) >= cap:
                return records
    return records


def profile_count(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return 1
    count = 0
    for _, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        count += sum(1 for name in files if Path(name).suffix.lower() in {".xml", ".json", ".txt", ".prof", ".db", ".sqlite", ".csv"})
        if count > 100000:
            break
    return count


def validate_assignments(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"present": False, "valid_schema": False}
    rows = read_csv(path)
    required = {"sha256", "host_id", "owner", "confidence", "is_sensitive", "is_critical", "attention_weight"}
    fields = set(rows[0]) if rows else set()
    if not required.issubset(fields):
        return {"present": True, "valid_schema": False, "rows": len(rows), "missing": sorted(required - fields)}
    sensitive = [r for r in rows if truthy(r.get("is_sensitive"))]
    known = [r for r in sensitive if normalize_owner(r.get("owner", "")) in {"APP", "TPL"}]
    critical = [r for r in sensitive if truthy(r.get("is_critical"))]
    critical_known = [r for r in critical if normalize_owner(r.get("owner", "")) in {"APP", "TPL"}]
    total_attention = sum(max(0.0, float(r.get("attention_weight") or 0)) for r in sensitive)
    known_attention = sum(max(0.0, float(r.get("attention_weight") or 0)) for r in known)
    apks = {r["sha256"] for r in sensitive}
    apk_tpl = {r["sha256"] for r in sensitive if normalize_owner(r.get("owner", "")) == "TPL" and str(r.get("library_id", "")).strip()}
    owners = Counter(normalize_owner(r.get("owner", "")) for r in sensitive)
    return {
        "present": True, "valid_schema": True, "rows": len(rows), "sensitive_rows": len(sensitive),
        "owner_counts": dict(owners),
        "sensitive_host_owner_coverage": len(known) / len(sensitive) if sensitive else 0.0,
        "critical_host_owner_coverage": len(critical_known) / len(critical) if critical else 0.0,
        "attention_owner_coverage": known_attention / total_attention if total_attention else 0.0,
        "apk_count": len(apks), "apk_with_concrete_tpl_identity": len(apk_tpl),
        "apk_concrete_tpl_fraction": len(apk_tpl) / len(apks) if apks else 0.0,
    }


def validate_truth(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"present": False, "valid_schema": False}
    rows = read_csv(path)
    required = {"truth_owner", "pred_owner", "high_confidence"}
    fields = set(rows[0]) if rows else set()
    if not required.issubset(fields):
        return {"present": True, "valid_schema": False, "rows": len(rows), "missing": sorted(required - fields)}
    selected = [r for r in rows if truthy(r.get("high_confidence"))]
    scored = [(normalize_owner(r["truth_owner"]), normalize_owner(r["pred_owner"])) for r in selected]
    correct = sum(int(a == b) for a, b in scored)
    return {
        "present": True, "valid_schema": True, "rows": len(rows), "high_confidence_rows": len(scored),
        "owner_precision": correct / len(scored) if scored else None,
        "correct": correct,
    }


def main() -> None:
    ensure_output()
    roots = candidate_roots()
    candidates = []
    for root in roots:
        candidates.extend(scan(root))

    explicit_tool_raw = os.environ.get("PASEA_TPL_TOOL", "").strip()
    explicit_profiles_raw = os.environ.get("PASEA_TPL_PROFILE_DB", "").strip()
    explicit_assignments_raw = os.environ.get("PASEA_OWNER_ASSIGNMENTS", "").strip()
    explicit_validation_raw = os.environ.get("PASEA_OWNER_VALIDATION", "").strip()
    tool = Path(explicit_tool_raw).resolve() if explicit_tool_raw else None
    profiles = Path(explicit_profiles_raw).resolve() if explicit_profiles_raw else None
    assignments = Path(explicit_assignments_raw).resolve() if explicit_assignments_raw else None
    validation = Path(explicit_validation_raw).resolve() if explicit_validation_raw else None

    result = {
        "status": "AUDITED",
        "future_rows_loaded": 0,
        "labels_used": False,
        "search_roots": [str(x) for x in roots],
        "candidate_count": len(candidates),
        "candidates": candidates[:1000],
        "commands": {name: shutil.which(name) for name in ["java", "docker", "jadx", "apktool", "python", "python3"]},
        "python_modules": {name: bool(importlib.util.find_spec(name)) for name in ["androguard", "networkx", "numpy", "pandas"]},
        "explicit": {
            "tool": str(tool) if tool else None,
            "tool_exists": bool(tool and tool.exists()),
            "profile_db": str(profiles) if profiles else None,
            "profile_db_exists": bool(profiles and profiles.exists()),
            "profile_count": profile_count(profiles) if profiles else 0,
            "assignments": str(assignments) if assignments else None,
            "validation": str(validation) if validation else None,
        },
        "assignment_audit": validate_assignments(assignments),
        "validation_audit": validate_truth(validation),
        "boundary": "Candidate filenames are inventory only. They are never accepted as verified TPL identity without explicit profiles, imported assignments, and independent owner ground truth.",
    }
    dump_json(OUTPUT / "environment_inventory.json", result)
    print(f"PASEA_TPL_ASSET_INVENTORY_DONE candidates={len(candidates)}")


if __name__ == "__main__":
    main()

