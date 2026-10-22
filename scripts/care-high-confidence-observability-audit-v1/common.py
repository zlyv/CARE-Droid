#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

PROJECT = Path(os.environ.get("HCOA_PROJECT", "/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT = Path(os.environ.get("HCOA_OUTPUT", str(PROJECT / "outputs/care-high-confidence-observability-audit-v1"))).resolve()


def ensure_output_boundary() -> None:
    root = (PROJECT / "outputs").resolve()
    if OUTPUT == root or root not in OUTPUT.parents or not OUTPUT.name.startswith("care-high-confidence-observability-audit-v"):
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: unsafe output {OUTPUT}")


def safe(path: Path) -> Path:
    ensure_output_boundary()
    path = Path(path).resolve()
    if path != OUTPUT and OUTPUT not in path.parents:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: write outside output {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def norm(value: Any) -> str:
    return str(value or "").strip().casefold()


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path = safe(path)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def dump_json(path: Path, value: Any) -> None:
    path = safe(path); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path, purpose: str) -> dict[str, Any]:
    path = Path(path).resolve(); stat = path.stat()
    return {"absolute_path": str(path), "purpose": purpose, "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": sha256_file(path)}


def first_existing(paths: Iterable[Path], label: str) -> Path:
    checked = []
    for path in paths:
        checked.append(str(path))
        if path.is_file():
            return path.resolve()
    raise RuntimeError(f"BLOCKED_BY_CONTRACT: missing {label}; checked={checked!r}")


def discover_assets() -> dict[str, Path]:
    return {
        "sample_audit": first_existing([
            PROJECT / "outputs/care-direct-c43-temporal-direction-audit-v1/sample_direction_audit.csv",
        ], "temporal sample direction audit"),
        "source_predictions": first_existing([
            PROJECT / "outputs/care-svm-c43-remove-bounded-correction-v1/source/U2_DIRECT_STAGE2/source_predictions.csv",
        ], "frozen source Direct-C43 predictions"),
        "future_predictions": first_existing([
            PROJECT / "outputs/care-svm-direct-c43-future-v1/predictions_label_free.csv",
        ], "frozen future Direct-C43 predictions"),
    }


PATH_FIELDS = ("absolute_path", "apk_path", "path", "file_path", "apk",
               "source_path", "sample_path", "input_path")
SHA_FIELDS = ("sha256", "sha", "sample_id", "id")


def _is_sha256(value: Any) -> bool:
    value = norm(value)
    return len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _row_pair(row: dict[str, Any]) -> tuple[str, str]:
    sid = next((norm(row.get(key)) for key in SHA_FIELDS if _is_sha256(row.get(key))), "")
    raw = next((str(row.get(key, "") or "").strip() for key in PATH_FIELDS
                if str(row.get(key, "") or "").strip().casefold().endswith(".apk")), "")
    if not sid and raw and _is_sha256(Path(raw).stem):
        sid = norm(Path(raw).stem)
    return sid, raw


def _candidate_pairs(path: Path):
    """Read the lineage/index formats already used by frozen future diagnostics."""
    suffix = "".join(path.suffixes).casefold()
    if suffix.endswith(".csv"):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if isinstance(row, dict):
                    yield _row_pair(row)
        return
    if suffix.endswith(".jsonl") or suffix.endswith(".jsonl.gz"):
        opener = gzip.open if suffix.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if isinstance(row, dict):
                        yield _row_pair(row)
        return
    if not suffix.endswith(".json"):
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        for row in data:
            if isinstance(row, dict):
                yield _row_pair(row)
        return
    if not isinstance(data, dict):
        return
    # Direct sha256 -> APK path maps are used by source extraction.
    for key, value in data.items():
        if _is_sha256(key) and isinstance(value, (str, Path)):
            yield norm(key), str(value)
    # Audit/lineage JSONs wrap rows under one of these keys.
    for key in ("records", "rows", "samples", "cohort", "data"):
        rows = data.get(key)
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    yield _row_pair(row)


def _resolve_apk(raw: str, index_path: Path) -> Path | None:
    candidate = Path(str(raw or "").strip())
    choices = [candidate] if candidate.is_absolute() else [PROJECT / candidate, index_path.parent / candidate]
    for choice in choices:
        if choice.is_file():
            return choice.resolve()
    return None


def load_apk_paths(wanted: set[str]) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    candidates: list[Path] = []
    for name in ("HCOA_SOURCE_APK_INDEX", "HCOA_FUTURE_APK_INDEX", "CCBC_FUTURE_APK_INDEX"):
        if os.environ.get(name):
            candidates.append(Path(os.environ[name]))
    candidates += [
        PROJECT / "outputs/care-ccbc-source-v3/apk_paths.json",
        PROJECT / "outputs/care-icde-source-v1/apk_asset_audit.json",
        PROJECT / "outputs/care-ccbc-diagnostics-v1/apk_paths.json",
        PROJECT / "outputs/care-ccbc-diagnostics-v1/future_apk_paths.json",
    ]
    # Reuse the same read-only future lineage discovery surface that produced
    # the completed 2018--2021 frozen diagnostics. Candidate names alone never
    # count as evidence: every parsed SHA and APK path is verified below.
    patterns = (
        "artifacts/care-p31rf-haig-source-v1/lineage/**/*",
        "artifacts/care-p31rf-haig-source-v1/**/*future*",
        "artifacts/care-future-local-assets-v2/**/*",
        "artifacts/**/*future*lineage*.csv",
        "artifacts/**/*apk*index*.csv",
    )
    seen = {str(path.resolve()) for path in candidates if path.exists()}
    for pattern in patterns:
        for path in PROJECT.glob(pattern):
            key = str(path.resolve())
            suffix = "".join(path.suffixes).casefold()
            if (path.is_file() and suffix.endswith((".csv", ".json", ".jsonl", ".jsonl.gz"))
                    and key not in seen):
                seen.add(key); candidates.append(path)
    result: dict[str, Path] = {}; audit = []
    for path in candidates:
        if not path.is_file():
            continue
        matched = 0
        try:
            for sid, raw in _candidate_pairs(path):
                sid = norm(sid)
                if sid not in wanted or sid in result or not raw:
                    continue
                apk = _resolve_apk(raw, path)
                if apk is not None:
                    result[sid] = apk; matched += 1
            if matched:
                audit.append({"path": str(path.resolve()), "status": "USED", "matched": matched})
            if len(result) == len(wanted):
                break
        except Exception as exc:
            audit.append({"path": str(path.resolve()), "status": "READ_ERROR", "error": repr(exc)})
    roots = []
    for name in ("HCOA_SOURCE_APK_ROOTS", "HCOA_FUTURE_APK_ROOTS", "CCBC_FUTURE_APK_ROOTS"):
        roots.extend(Path(value) for value in os.environ.get(name, "").split(os.pathsep) if value)
    for root in roots:
        matched = 0
        if root.is_dir():
            for apk in root.rglob("*.apk"):
                sid = apk.stem.casefold()
                if sid in wanted and sid not in result:
                    result[sid] = apk.resolve(); matched += 1
        audit.append({"path": str(root), "root_scan_matched": matched})
    return result, audit


def exact_mcnemar_p(b: int, c: int) -> float:
    n = int(b + c)
    if n == 0:
        return 1.0
    low = min(int(b), int(c))
    tail = sum(math.comb(n, k) for k in range(low + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)
