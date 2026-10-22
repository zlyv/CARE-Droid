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

PROJECT = Path(os.environ.get("SDC43F_PROJECT", "/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT = Path(os.environ.get("SDC43F_OUTPUT", str(PROJECT / "outputs/care-svm-direct-c43-future-v1"))).resolve()
FAMILY_SCRIPT = PROJECT / "scripts/care-p31-classifier-family-stage2-v1"
FAMILY_OUTPUT = PROJECT / "outputs/care-p31-classifier-family-stage2-v1"
DIRECT_SCRIPT = PROJECT / "scripts/care-svm-c43-remove-bounded-correction-v1"
DIRECT_OUTPUT = PROJECT / "outputs/care-svm-c43-remove-bounded-correction-v1"
FORMAL_DIAG = PROJECT / "artifacts/care-droid-formal-diagnostics-v1"
DIRECT_METHOD = "P31SVM__SVMRoute__A0_ROLE_C43_RQ__DirectStage2"
YEARS = (2018, 2019, 2020, 2021)


def norm(value: Any) -> str:
    return str(value or "").strip().casefold()


def safe_output(path: Path) -> Path:
    path = Path(path).resolve()
    try:
        path.relative_to(OUTPUT)
    except ValueError as exc:
        raise RuntimeError("BLOCKED_BY_CONTRACT: output outside new diagnostic directory") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows) -> None:
    rows = list(rows); path = safe_output(path)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def dump_json(path: Path, value: Any) -> None:
    path = safe_output(path); temporary = path.with_suffix(path.suffix + ".tmp")
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


def auc(y, score) -> float:
    y = np.asarray(y, int); score = np.asarray(score, float)
    positive = int(y.sum()); negative = len(y) - positive
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort"); ordered = score[order]
    ranks = np.empty(len(score), float); start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and ordered[end] == ordered[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2; start = end
    return float((ranks[y == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def metrics(y, base, final, route) -> dict[str, Any]:
    y = np.asarray(y, int); base = np.asarray(base, float); final = np.asarray(final, float)
    route = np.asarray(route, bool); base_pred = base >= .5; pred = final >= .5
    before = base_pred == y; after = pred == y; fixed = (~before) & after; broken = before & (~after)
    tn = int(np.sum((y == 0) & (~pred))); fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & (~pred))); tp = int(np.sum((y == 1) & pred))
    return {"n": len(y), "BA": float((tn / max(tn + fp, 1) + tp / max(tp + fn, 1)) / 2),
            "AUC": auc(y, final), "F1": float(2 * tp / max(2 * tp + fp + fn, 1)),
            "accuracy": float(np.mean(pred == y)), "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "errors": fp + fn, "fixed": int(fixed.sum()), "broken": int(broken.sum()),
            "net": int(fixed.sum() - broken.sum()), "fixed_FN": int(np.sum(fixed & (y == 1))),
            "fixed_FP": int(np.sum(fixed & (y == 0))), "broken_TN": int(np.sum(broken & (y == 0))),
            "broken_TP": int(np.sum(broken & (y == 1))), "routed": int(route.sum()),
            "routed_stage1_errors": int(np.sum(route & (~before))),
            "route_precision": float(np.sum(route & (~before)) / max(route.sum(), 1)),
            "error_recall": float(np.sum(route & (~before)) / max((~before).sum(), 1)),
            "class_flips": int(np.sum(base_pred != pred))}


def mcnemar(candidate_ok, baseline_ok):
    candidate_ok = np.asarray(candidate_ok, bool); baseline_ok = np.asarray(baseline_ok, bool)
    candidate_only = int(np.sum(candidate_ok & (~baseline_ok)))
    baseline_only = int(np.sum((~candidate_ok) & baseline_ok)); n = candidate_only + baseline_only
    if n == 0:
        return candidate_only, baseline_only, 1.0
    tail = sum(math.comb(n, k) for k in range(min(candidate_only, baseline_only) + 1)) / (2 ** n)
    return candidate_only, baseline_only, float(min(1.0, 2 * tail))
