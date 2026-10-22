#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(os.environ.get("CMW_PROJECT", "/home/boyuanjie/projects/tasa-droid")).resolve()
SOURCE_OUTPUT = Path(os.environ.get(
    "CMW_SOURCE_OUTPUT", PROJECT / "outputs/care-contextual-multihop-witness-v1"
)).resolve()
OUTPUT = Path(os.environ.get(
    "CMW_POSTGATE_OUTPUT", PROJECT / "outputs/care-cmw-postgate-source-audit-v1"
)).resolve()
FORMAL = Path(os.environ.get(
    "CMW_FORMAL", PROJECT / "artifacts/care-p31rf-c43-haig-host-ablation-source-v1/arms/A0_ROLE_ONLY/source_predictions.csv"
)).resolve()
FEATURE_DIR = SOURCE_OUTPUT / "cache/features/source"
SEED = 20261008
ARMS = ("REAL", "ONE_HOP", "ROOT_SHUFFLE", "CALLSITE_SHUFFLE", "COUNT_ONLY", "PATH_ONLY")
COUNT_FIELDS = ("graph_method_count", "graph_edge_count", "root_count", "host_count")


def norm(value: Any) -> str:
    return str(value or "").strip().casefold()


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, purpose: str) -> dict[str, Any]:
    path = Path(path).resolve(); stat = path.stat()
    return {"absolute_path": str(path), "purpose": purpose, "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": sha256_file(path)}


def binary_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, int); score = np.asarray(score, float)
    positive, negative = int(y.sum()), len(y) - int(y.sum())
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float); sorted_score = score[order]; start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_score[end] == sorted_score[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((ranks[y == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def parse_formal() -> list[dict[str, Any]]:
    raw = read_csv(FORMAL)
    if len(raw) != 799:
        raise RuntimeError(f"source rows changed: {len(raw)}")
    method = "P31RF__RFRoute__A0_ROLE_ONLY__Bounded"
    output = []
    for row in raw:
        base = float(row["Global24__prob"])
        output.append({"sha256": norm(row["sha256"]), "fold": int(float(row["fold"])),
                       "label": int(float(row["label"])), "stage1_prob": base,
                       "stage1_pred": int(base >= .5), "routed": int(float(row["A25__routed"])),
                       "formal_prob": float(row[method + "__prob"])})
    return output


def bootstrap_auc_difference(y: np.ndarray, real: np.ndarray, comparator: np.ndarray,
                             seed: int, repeats: int = 2000) -> dict[str, Any]:
    y = np.asarray(y, int); real = np.asarray(real, float); comparator = np.asarray(comparator, float)
    positive = np.flatnonzero(y == 1); negative = np.flatnonzero(y == 0)
    if not len(positive) or not len(negative):
        return {"repeats": 0, "difference": float("nan"), "ci95": [float("nan"), float("nan")]}
    rng = np.random.default_rng(seed); differences = np.empty(repeats, float)
    for index in range(repeats):
        sample = np.concatenate((rng.choice(positive, len(positive), replace=True),
                                 rng.choice(negative, len(negative), replace=True)))
        differences[index] = binary_auc(y[sample], real[sample]) - binary_auc(y[sample], comparator[sample])
    return {"repeats": repeats, "difference": binary_auc(y, real) - binary_auc(y, comparator),
            "ci95": [float(np.quantile(differences, .025)), float(np.quantile(differences, .975))],
            "probability_difference_gt_zero": float(np.mean(differences > 0))}
