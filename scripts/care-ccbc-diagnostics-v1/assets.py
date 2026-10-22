#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import json
import os
from pathlib import Path
from typing import Any

from common import (
    OUTPUT, PROJECT, discover_compact_cache, discover_endpoint_cache,
    discover_formal_predictions, discover_formal_script_files, discover_p31,
    discover_role_manifest, file_record, norm,
)


PATH_FIELDS = (
    "absolute_path", "apk_path", "path", "file_path", "apk",
    "source_path", "sample_path", "input_path",
)
SHA_FIELDS = ("sha256", "sha", "sample_id", "id")


def _index_candidates() -> list[Path]:
    result = []
    explicit = os.environ.get("CCBC_APK_INDEX")
    if explicit:
        result.append(Path(explicit))
    result.append(PROJECT / "artifacts/care-p31rf-haig-source-v1/lineage/source_2017.csv")
    patterns = (
        "artifacts/care-p31rf-haig-source-v1/**/*",
        "artifacts/care-source-local-assets-v2/**/*",
        "artifacts/care-litelocal-p31-diagnostics-v1/**/*",
        "artifacts/**/*apk*index*.csv",
        "artifacts/**/*cohort*.csv",
        "artifacts/**/*manifest*.csv",
    )
    seen = set()
    for pattern in patterns:
        for path in PROJECT.glob(pattern):
            if path.is_file() and path.suffix.casefold() in {".csv", ".json", ".jsonl", ".gz"}:
                key = str(path.resolve())
                if key not in seen:
                    seen.add(key); result.append(path)
    return result


def _path_from_row(row: dict[str, Any]) -> str:
    for key in PATH_FIELDS:
        value = str(row.get(key, "") or "").strip()
        if value.lower().endswith(".apk"):
            return value
    return ""


def _sha_from_row(row: dict[str, Any]) -> str:
    for key in SHA_FIELDS:
        value = norm(row.get(key))
        if len(value) == 64 and all(ch in "0123456789abcdef" for ch in value):
            return value
    return ""


def _rows(path: Path):
    suffixes = "".join(path.suffixes).casefold()
    if suffixes.endswith(".csv"):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            yield from csv.DictReader(handle)
    elif suffixes.endswith(".json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            yield from value
        elif isinstance(value, dict):
            for key in ("rows", "samples", "cohort", "data"):
                if isinstance(value.get(key), list):
                    yield from value[key]; break
    elif suffixes.endswith(".jsonl") or suffixes.endswith(".jsonl.gz"):
        opener = gzip.open if suffixes.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def discover_apk_map(wanted: set[str]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    mapping: dict[str, str] = {}
    audit = []
    for path in _index_candidates():
        found = 0
        try:
            for row in _rows(path):
                if not isinstance(row, dict):
                    continue
                sid = _sha_from_row(row); apk = _path_from_row(row)
                if sid in wanted and apk:
                    resolved = Path(apk)
                    if not resolved.is_absolute():
                        resolved = (PROJECT / resolved).resolve()
                    if resolved.is_file():
                        mapping.setdefault(sid, str(resolved)); found += 1
        except Exception as exc:
            audit.append({"path": str(path), "status": "READ_ERROR", "error": repr(exc)})
            continue
        if found:
            audit.append({"path": str(path.resolve()), "status": "USED", "matched_rows": found})
        if len(mapping) == len(wanted):
            break
    roots = [Path(value) for value in os.environ.get("CCBC_APK_ROOTS", "").split(os.pathsep) if value]
    if len(mapping) < len(wanted) and roots:
        for root in roots:
            if not root.is_dir():
                audit.append({"path": str(root), "status": "ROOT_MISSING"}); continue
            found = 0
            for path in root.rglob("*.apk"):
                sid = norm(path.stem)
                if sid in wanted and sid not in mapping:
                    mapping[sid] = str(path.resolve()); found += 1
                    if len(mapping) == len(wanted):
                        break
            audit.append({"path": str(root.resolve()), "status": "SCANNED_ROOT", "matched_rows": found})
            if len(mapping) == len(wanted):
                break
    return mapping, audit


def load_endpoint_rows(wanted: set[str]) -> dict[str, dict[str, Any]]:
    path = discover_endpoint_cache()
    result = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = norm(row.get("sha256", row.get("sample_id")))
            if sid not in wanted:
                continue
            if str(row.get("status", "OK")).upper() not in {"OK", "SUCCESS"}:
                continue
            payload = row.get("payload", row)
            result[sid] = payload
            if len(result) == len(wanted):
                break
    return result


def baseline_assets() -> list[tuple[Path, str]]:
    assets = [
        (discover_formal_predictions(), "frozen formal predictions"),
        (discover_p31(), "P31 source cache"),
        (discover_endpoint_cache(), "frozen endpoint/call-graph cache"),
        (discover_compact_cache(), "frozen formal Host cache"),
        (discover_role_manifest(), "frozen role mapping manifest"),
    ]
    assets.extend(discover_formal_script_files())
    unique = {}
    for path, purpose in assets:
        unique[str(Path(path).resolve())] = (Path(path).resolve(), purpose)
    return list(unique.values())


def manifest_records() -> list[dict[str, Any]]:
    return [file_record(path, purpose) for path, purpose in baseline_assets()]


