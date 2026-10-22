#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(os.environ.get("SCSCD_PROJECT", "/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT = Path(os.environ.get("SCSCD_OUTPUT", str(PROJECT / "outputs/care-svm-consensus-selective-correction-diagnostics-v1"))).resolve()
STAGE2 = PROJECT / "outputs/care-p31-classifier-family-stage2-v1"
CLASSIFIERS = PROJECT / "outputs/care-p31-classifier-family-v1"
FORMAL_DIAG = PROJECT / "artifacts/care-droid-formal-diagnostics-v1"
SOURCE_AUDIT = PROJECT / "outputs/care-svm-consensus-selective-correction-v1"
FROZEN_STAGE2 = STAGE2 / "frozen_future_stage2_predictions.csv"
WITHIN_STAGE2 = STAGE2 / "same_year_stage2_predictions.csv"
FROZEN_MEMBERS = FORMAL_DIAG / "references/future_label_free.csv"
WITHIN_MEMBERS = CLASSIFIERS / "within_year_predictions.csv"
SEED = 20261009
YEARS = (2018, 2019, 2020, 2021)
ARMS = ("S0_STAGE1_SVM", "S1_ALL_C43", "S2_MAJORITY_2_OF_3", "S3_UNANIMOUS_3_OF_3")


def norm(value: Any) -> str:
    return str(value or "").strip().casefold()


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows) -> None:
    rows = list(rows); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def dump(path: Path, value: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


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


def binary_auc(y, score) -> float:
    y = np.asarray(y, int); score = np.asarray(score, float)
    pos = int(y.sum()); neg = len(y) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort"); ordered = score[order]
    ranks = np.empty(len(score), float); start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and ordered[end] == ordered[start]: end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2; start = end
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def metrics(y, base, final, route) -> dict[str, Any]:
    y = np.asarray(y, int); base = np.asarray(base, float); final = np.asarray(final, float); route = np.asarray(route, bool)
    bp = base >= .5; pred = final >= .5; before = bp == y; after = pred == y
    tn = int(np.sum((y == 0) & (~pred))); fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & (~pred))); tp = int(np.sum((y == 1) & pred))
    fixed = (~before) & after; broken = before & (~after)
    return {"n": len(y), "BA": float((tn / max(tn + fp, 1) + tp / max(tp + fn, 1)) / 2),
        "AUC": binary_auc(y, final), "F1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp, "errors": fp + fn,
        "fixed": int(fixed.sum()), "broken": int(broken.sum()), "net": int(fixed.sum() - broken.sum()),
        "fixed_FN": int(np.sum(fixed & (y == 1))), "fixed_FP": int(np.sum(fixed & (y == 0))),
        "broken_TN": int(np.sum(broken & (y == 0))), "broken_TP": int(np.sum(broken & (y == 1))),
        "routed": int(route.sum()), "class_flips": int(np.sum(bp != pred))}
