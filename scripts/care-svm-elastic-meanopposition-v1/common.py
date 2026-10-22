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

PROJECT = Path(os.environ.get("EMO_PROJECT", "/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT = Path(os.environ.get("EMO_OUTPUT", str(PROJECT / "outputs/care-svm-elastic-meanopposition-v1"))).resolve()
DIRECT_METHOD = "P31SVM__SVMRoute__A0_ROLE_C43_RQ__DirectStage2"
YEARS = (2018, 2019, 2020, 2021)


def ensure_output(path: Path | None = None) -> Path:
    root = (PROJECT / "outputs").resolve()
    if OUTPUT == root or root not in OUTPUT.parents or not OUTPUT.name.startswith("care-svm-elastic-meanopposition-v"):
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: unsafe output {OUTPUT}")
    target = OUTPUT if path is None else Path(path).resolve()
    if target != OUTPUT and OUTPUT not in target.parents:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: write outside output {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    rows = list(rows); path = ensure_output(path)
    fields: list[str] = []
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
    path = ensure_output(path); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def norm(value: Any) -> str:
    return str(value or "").strip().casefold()


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


def discover() -> dict[str, Path]:
    candidates = {
        "source": PROJECT / "outputs/care-svm-c43-remove-bounded-correction-v1/source/U2_DIRECT_STAGE2/source_predictions.csv",
        "architecture": PROJECT / "outputs/care-svm-c43-remove-bounded-correction-v1/source/U2_DIRECT_STAGE2/architecture.json",
        "future": PROJECT / "outputs/care-svm-direct-c43-future-v1/predictions_label_free.csv",
        "future_labels": PROJECT / "artifacts/care-droid-formal-diagnostics-v1/frozen_predictions_with_labels.csv",
    }
    overrides = {"source": "EMO_SOURCE", "architecture": "EMO_ARCHITECTURE", "future": "EMO_FUTURE", "future_labels": "EMO_FUTURE_LABELS"}
    result = {}
    for key, default in candidates.items():
        path = Path(os.environ.get(overrides[key], str(default))).resolve()
        if not path.is_file():
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: missing {key}: {path}")
        result[key] = path
    return result


def top_k(score, sha, group, fraction: float) -> np.ndarray:
    score = np.asarray(score, float); sha = np.asarray(sha).astype(str); group = np.asarray(group)
    route = np.zeros(len(score), bool)
    for value in sorted(set(group.tolist())):
        index = np.flatnonzero(group == value); count = int(round(len(index) * fraction))
        order = np.lexsort((sha[index], -score[index])); route[index[order[:count]]] = True
    return route


def cap_route(candidate, score, sha, group, fraction: float) -> np.ndarray:
    candidate = np.asarray(candidate, bool); score = np.asarray(score, float)
    sha = np.asarray(sha).astype(str); group = np.asarray(group); route = np.zeros(len(score), bool)
    for value in sorted(set(group.tolist())):
        index = np.flatnonzero(group == value); local = index[candidate[index]]
        count = min(len(local), int(round(len(index) * fraction)))
        order = np.lexsort((sha[local], -score[local])); route[local[order[:count]]] = True
    return route


def threshold_higher(values, quantile: float = .75) -> float:
    values = np.asarray(values, float)
    try:
        return float(np.quantile(values, quantile, method="higher"))
    except TypeError:
        return float(np.quantile(values, quantile, interpolation="higher"))


def auc(y, score) -> float:
    y = np.asarray(y, int); score = np.asarray(score, float)
    positive = int(y.sum()); negative = len(y) - positive
    if not positive or not negative:
        return float("nan")
    order = np.argsort(score, kind="mergesort"); ordered = score[order]
    ranks = np.empty(len(y), float); start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and ordered[end] == ordered[start]: end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2; start = end
    return float((ranks[y == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def metrics(y, base, expert, route) -> dict[str, Any]:
    y = np.asarray(y, int); base = np.asarray(base, float); expert = np.asarray(expert, float); route = np.asarray(route, bool)
    final = np.where(route, expert, base); base_pred = base >= .5; pred = final >= .5
    before = base_pred == y; after = pred == y; fixed = (~before) & after; broken = before & (~after)
    tn = int(np.sum((y == 0) & (~pred))); fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & (~pred))); tp = int(np.sum((y == 1) & pred))
    captured = int(np.sum(route & (~before))); routed_correct = int(np.sum(route & before))
    return {"n": len(y), "BA": float((tn / max(tn + fp, 1) + tp / max(tp + fn, 1)) / 2),
            "AUC": auc(y, final), "F1": float(2 * tp / max(2 * tp + fp + fn, 1)),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp, "errors": fp + fn,
            "fixed": int(fixed.sum()), "broken": int(broken.sum()), "net": int(fixed.sum() - broken.sum()),
            "fixed_FN": int(np.sum(fixed & (y == 1))), "fixed_FP": int(np.sum(fixed & (y == 0))),
            "broken_TN": int(np.sum(broken & (y == 0))), "broken_TP": int(np.sum(broken & (y == 1))),
            "routed": int(route.sum()), "routed_stage1_errors": captured,
            "routed_stage1_correct": routed_correct,
            "error_recall": float(captured / max((~before).sum(), 1)),
            "route_precision": float(captured / max(route.sum(), 1)),
            "class_flips": int(np.sum(base_pred != pred))}


def mcnemar(candidate_ok, baseline_ok) -> tuple[int, int, float]:
    candidate_ok = np.asarray(candidate_ok, bool); baseline_ok = np.asarray(baseline_ok, bool)
    a = int(np.sum(candidate_ok & (~baseline_ok))); b = int(np.sum((~candidate_ok) & baseline_ok)); n = a + b
    if not n: return a, b, 1.0
    tail = sum(math.comb(n, k) for k in range(min(a, b) + 1)) / (2 ** n)
    return a, b, float(min(1.0, 2 * tail))
