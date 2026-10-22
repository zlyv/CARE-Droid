#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

PROJECT = Path(os.environ.get("SICA_PROJECT", "/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT = Path(os.environ.get(
    "SICA_OUTPUT", str(PROJECT / "outputs/care-static-identifiability-ceiling-audit-v1")
)).resolve()


def ensure_output_boundary() -> None:
    root = (PROJECT / "outputs").resolve()
    if OUTPUT == root or root not in OUTPUT.parents or not OUTPUT.name.startswith("care-static-identifiability-ceiling-audit-v"):
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: unsafe output {OUTPUT}")


def safe(path: Path) -> Path:
    ensure_output_boundary(); path = Path(path).resolve()
    if path != OUTPUT and OUTPUT not in path.parents:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: write outside output {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def norm(value: Any) -> str:
    return str(value or "").strip().casefold()


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows); path = safe(path); fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def dump_json(path: Path, value: Any) -> None:
    path = safe(path); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def first_existing(paths: Iterable[Path], label: str) -> Path:
    checked = []
    for path in paths:
        checked.append(str(path))
        if path.is_file() and path.stat().st_size:
            return path.resolve()
    raise RuntimeError(f"BLOCKED_BY_CONTRACT: missing {label}; checked={checked!r}")


def discover_assets() -> dict[str, Path]:
    return {
        "temporal": first_existing([
            PROJECT / "outputs/care-direct-c43-temporal-direction-audit-v1/sample_direction_audit.csv",
        ], "2017-2021 frozen temporal audit"),
        "source_family": first_existing([
            PROJECT / "outputs/care-p31-classifier-family-v1/source_predictions.csv",
        ], "source classifier-family predictions"),
        "future_family": first_existing([
            PROJECT / "outputs/care-p31-classifier-family-v1/frozen_future_predictions.csv",
        ], "future classifier-family predictions"),
        "neighbors": first_existing([
            PROJECT / "outputs/care-svm-c43-representation-collision-audit-v1/neighbor_samples.csv",
        ], "source OOF representation-neighbor audit"),
        "collision": first_existing([
            PROJECT / "outputs/care-svm-c43-representation-collision-audit-v1/collision_audit.json",
        ], "source representation-collision decision"),
        "hcoa": first_existing([
            PROJECT / "outputs/care-high-confidence-observability-audit-v1/final_decision.json",
        ], "high-confidence observability decision"),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def asset_record(path: Path, purpose: str) -> dict[str, Any]:
    path = Path(path).resolve(); status = path.stat()
    return {"absolute_path": str(path), "purpose": purpose, "size_bytes": status.st_size,
            "mtime_ns": status.st_mtime_ns, "sha256": sha256_file(path)}


def fraction(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0
