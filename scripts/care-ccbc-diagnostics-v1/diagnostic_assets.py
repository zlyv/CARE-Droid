#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from common import OUTPUT, PROJECT, norm, read_csv, read_jsonl_gz, sha256_file


SOURCE_OUTPUT = Path(os.environ.get(
    "CCBC_SOURCE_OUTPUT", PROJECT / "outputs/care-ccbc-source-v3"
)).resolve()
FORMAL_ROOT = Path(os.environ.get(
    "CCBC_FORMAL_DIAGNOSTICS", PROJECT / "artifacts/care-droid-formal-diagnostics-v1"
)).resolve()
EXPECTED_PROTOCOL_HASH = "1940d7d426777fb8a8450c38dbc6cf2958067e8bb69d0b4b214c65986accf5be"
YEARS = (2018, 2019, 2020, 2021)
YEAR_COUNTS = {2018: 793, 2019: 399, 2020: 397, 2021: 400}


def require(path: Path, label: str) -> Path:
    path = Path(path)
    if not path.is_file() or not path.stat().st_size:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: missing {label}: {path}")
    return path.resolve()


def load_formal_common():
    path = require(PROJECT / "scripts/care-droid-formal-diagnostics-v1/common.py", "formal diagnostics common")
    spec = importlib.util.spec_from_file_location("ccbc_formal_common", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def resolved_inputs() -> dict[str, Any]:
    return json.loads(require(FORMAL_ROOT / "resolved_inputs.json", "formal resolved inputs").read_text(encoding="utf-8"))


def source_protocol_check() -> dict[str, Any]:
    value = require(SOURCE_OUTPUT / "protocol_hash.txt", "frozen source protocol hash").read_text(encoding="utf-8").strip()
    if value != EXPECTED_PROTOCOL_HASH:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: source protocol hash changed {value}")
    summary = json.loads(require(SOURCE_OUTPUT / "final_summary.json", "source final summary").read_text(encoding="utf-8"))
    return {"protocol_hash": value, "source_summary": summary}


def future_cohort() -> list[tuple[str, int]]:
    formal = load_formal_common()
    _, rows = formal.load_future_cohort()
    output = [(norm(sid), int(year)) for sid, year in rows]
    counts = {year: sum(value == year for _, value in output) for year in YEARS}
    if len(output) != 1989 or counts != YEAR_COUNTS:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: future cohort changed n={len(output)} counts={counts}")
    return output


def future_labels() -> tuple[Path, dict[str, int]]:
    formal = load_formal_common()
    path, labels = formal.load_future_labels()
    labels = {norm(key): int(value) for key, value in labels.items()}
    cohort = future_cohort()
    if any(sid not in labels for sid, _ in cohort):
        raise RuntimeError("BLOCKED_BY_CONTRACT: incomplete diagnostic future labels")
    return Path(path).resolve(), labels


def future_fold_rows() -> list[dict[str, Any]]:
    formal = load_formal_common()
    sources, raw = formal.discover_fold_assignment_rows()
    rows, mismatches = formal.canonicalize_fold_assignments(raw_rows=raw)
    if len(rows) != 1989:
        raise RuntimeError("BLOCKED_BY_CONTRACT: future fold assignment count changed")
    return [{"sha256": norm(row["sample_id"]), "year": int(row["year"]), "fold": int(row["fold"])} for row in rows]


def future_p31_path() -> Path:
    resolved = resolved_inputs()
    return require(Path(resolved["future_p31_cache"]), "future P31 cache")


def endpoint_cache_path() -> Path:
    resolved = resolved_inputs()
    return require(Path(resolved["endpoint_cache"]), "2788-row endpoint cache")


def a0_region_cache_path() -> Path:
    candidates = [
        FORMAL_ROOT / "cache/a0_role_2017_2021.npz",
        Path(os.environ["CCBC_FUTURE_A0_CACHE"]) if os.environ.get("CCBC_FUTURE_A0_CACHE") else Path("__missing__"),
    ]
    for path in candidates:
        if path.is_file() and path.stat().st_size:
            return path.resolve()
    raise RuntimeError("BLOCKED_BY_CONTRACT: missing formal 2017-2021 A0 Host cache")


def frozen_formal_rows() -> list[dict[str, str]]:
    candidates = [
        FORMAL_ROOT / "frozen_predictions_with_labels.csv",
        FORMAL_ROOT / "frozen/future_predictions_label_free.csv",
    ]
    path = next((value for value in candidates if value.is_file() and value.stat().st_size), None)
    if path is None:
        raise RuntimeError("BLOCKED_BY_CONTRACT: frozen formal future predictions missing")
    rows = read_csv(path)
    _, labels = future_labels()
    for row in rows:
        sid = norm(row.get("sha256")); row["sha256"] = sid
        if not str(row.get("label", "")).strip(): row["label"] = str(labels[sid])
    if len(rows) != 1989:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: frozen future rows={len(rows)}")
    return rows


def within_formal_rows() -> list[dict[str, str]]:
    rows = []
    for path in (
        FORMAL_ROOT / "within_2018_2019/predictions.csv",
        FORMAL_ROOT / "within_2020_2021/predictions.csv",
    ):
        rows.extend(read_csv(require(path, "formal same-year predictions")))
    for row in rows: row["sha256"] = norm(row.get("sha256"))
    if len(rows) != 1989:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: same-year rows={len(rows)}")
    return rows


PATH_FIELDS = ("absolute_path", "apk_path", "path", "file_path", "apk", "source_path", "sample_path", "input_path")
SHA_FIELDS = ("sha256", "sha", "sample_id", "id")


def _path_from_row(row: dict[str, Any]) -> str:
    for key in PATH_FIELDS:
        value = str(row.get(key, "") or "").strip()
        if value.casefold().endswith(".apk"): return value
    return ""


def _sha_from_row(row: dict[str, Any]) -> str:
    for key in SHA_FIELDS:
        value = norm(row.get(key))
        if len(value) == 64 and all(ch in "0123456789abcdef" for ch in value): return value
    return ""


def _candidate_rows(path: Path):
    suffix = "".join(path.suffixes).casefold()
    if suffix.endswith(".csv"):
        with path.open("r", encoding="utf-8-sig", newline="") as handle: yield from csv.DictReader(handle)
    elif suffix.endswith(".json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list): yield from value
        elif isinstance(value, dict):
            for key in ("rows", "samples", "cohort", "data"):
                if isinstance(value.get(key), list): yield from value[key]; break
    elif suffix.endswith(".jsonl") or suffix.endswith(".jsonl.gz"):
        opener = gzip.open if suffix.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip(): yield json.loads(line)


def discover_future_apks() -> tuple[dict[str, str], list[dict[str, Any]]]:
    wanted = {sid for sid, _ in future_cohort()}
    explicit = os.environ.get("CCBC_FUTURE_APK_INDEX")
    candidates = [Path(explicit)] if explicit else []
    patterns = (
        "artifacts/care-p31rf-haig-source-v1/lineage/**/*",
        "artifacts/care-p31rf-haig-source-v1/**/*future*",
        "artifacts/care-future-local-assets-v2/**/*",
        "artifacts/**/*future*lineage*.csv",
        "artifacts/**/*apk*index*.csv",
    )
    seen = set()
    for pattern in patterns:
        for path in PROJECT.glob(pattern):
            key = str(path.resolve())
            if path.is_file() and path.suffix.casefold() in {".csv", ".json", ".jsonl", ".gz"} and key not in seen:
                seen.add(key); candidates.append(path)
    mapping: dict[str, str] = {}; audit = []
    for path in candidates:
        found = 0
        try:
            for row in _candidate_rows(path):
                if not isinstance(row, dict): continue
                sid, raw = _sha_from_row(row), _path_from_row(row)
                if sid not in wanted or not raw: continue
                resolved = Path(raw)
                if not resolved.is_absolute(): resolved = (PROJECT / resolved).resolve()
                if resolved.is_file(): mapping.setdefault(sid, str(resolved)); found += 1
        except Exception as exc:
            audit.append({"path": str(path), "status": "READ_ERROR", "error": repr(exc)}); continue
        if found: audit.append({"path": str(path.resolve()), "status": "USED", "matched_rows": found})
        if len(mapping) == len(wanted): break
    roots = [Path(value) for value in os.environ.get("CCBC_FUTURE_APK_ROOTS", "").split(os.pathsep) if value]
    for root in roots:
        if len(mapping) == len(wanted): break
        if not root.is_dir(): audit.append({"path": str(root), "status": "ROOT_MISSING"}); continue
        found = 0
        for path in root.rglob("*.apk"):
            sid = norm(path.stem)
            if sid in wanted and sid not in mapping: mapping[sid] = str(path.resolve()); found += 1
        audit.append({"path": str(root.resolve()), "status": "SCANNED_ROOT", "matched_rows": found})
    return mapping, audit


def load_future_endpoint_rows(wanted: set[str]) -> dict[str, dict[str, Any]]:
    result = {}
    with gzip.open(endpoint_cache_path(), "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip(): continue
            row = json.loads(line); sid = norm(row.get("sha256", row.get("sample_id")))
            if sid not in wanted or str(row.get("status", "OK")).upper() not in {"OK", "SUCCESS"}: continue
            result[sid] = row.get("payload", row)
            if len(result) == len(wanted): break
    return result


def load_attention(path: Path) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    with np.load(require(path, "attention cache"), allow_pickle=False) as z:
        ids = [norm(value) for value in z["sha256"]]; offsets = np.asarray(z["offsets"], int)
        flat = np.asarray(z["attention"], float); critical = np.asarray(z["critical_index"], int)
    attention = {sid: flat[offsets[i]:offsets[i + 1]] for i, sid in enumerate(ids)}
    return attention, dict(zip(ids, critical.tolist()))

