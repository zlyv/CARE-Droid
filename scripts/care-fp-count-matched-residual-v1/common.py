#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(os.environ.get("FPR_PROJECT", "/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT = Path(os.environ.get("FPR_OUTPUT", str(PROJECT / "outputs/care-fp-count-matched-residual-v1"))).resolve()
SCS_OUTPUT = Path(os.environ.get("FPR_SCS_OUTPUT", str(PROJECT / "outputs/care-sensitive-consequence-slice-v1"))).resolve()
SEED = 20261008


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, purpose: str) -> dict[str, Any]:
    path = Path(path).resolve(); status = path.stat()
    return {"absolute_path": str(path), "purpose": purpose, "size_bytes": status.st_size,
            "mtime_ns": status.st_mtime_ns, "sha256": sha256_file(path)}


def formal_path() -> Path:
    candidates = [Path(os.environ["FPR_FORMAL"])] if os.environ.get("FPR_FORMAL") else []
    candidates += [PROJECT / "artifacts/care-p31rf-c43-haig-host-ablation-source-v1/arms/A0_ROLE_ONLY/source_predictions.csv",
                   PROJECT / "artifacts/care-droid-formal-v1/formal_predictions.csv"]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise RuntimeError("BLOCKED_BY_CONTRACT: formal source predictions missing")


def parse_formal() -> list[dict[str, Any]]:
    raw = read_csv(formal_path())
    if len(raw) != 799:
        raise RuntimeError(f"source rows changed: {len(raw)}")
    base_field = "Global24__prob" if "Global24__prob" in raw[0] else "stage1_prob"
    route_field = "A25__routed" if "A25__routed" in raw[0] else "routed"
    rows = []
    for item in raw:
        probability = float(item[base_field])
        rows.append({"sha256": str(item["sha256"]).strip().casefold(), "fold": int(float(item["fold"])),
                     "label": int(float(item["label"])), "stage1_prob": probability,
                     "stage1_pred": int(probability >= .5), "routed": int(float(item[route_field]))})
    return rows


def binary_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, int); score = np.asarray(score, float); positive = int(y.sum()); negative = len(y) - positive
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort"); ranks = np.empty(len(score), float); ordered = score[order]; start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and ordered[end] == ordered[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2; start = end
    return float((ranks[y == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def spearman(left, right) -> float:
    left, right = np.asarray(left, float), np.asarray(right, float)
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    try:
        from scipy.stats import spearmanr
        value = float(spearmanr(left, right).statistic)
        return 0.0 if not math.isfinite(value) else value
    except Exception:
        return float(np.corrcoef(np.argsort(np.argsort(left)), np.argsort(np.argsort(right)))[0, 1])
