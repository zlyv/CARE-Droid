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

PROJECT = Path(os.environ.get("PCF_PROJECT", "/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT = Path(os.environ.get("PCF_OUTPUT", str(PROJECT / "outputs/care-p31-classifier-family-v1"))).resolve()
FORMAL_DIAG = PROJECT / "artifacts/care-droid-formal-diagnostics-v1"
SOURCE_MEMBER = PROJECT / "artifacts/care-litelocal-stage1-attribution-source-v1/source_member_oof_predictions.csv"
SOURCE_P31 = PROJECT / "artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz"
FUTURE_MEMBER = FORMAL_DIAG / "references/future_label_free.csv"
FUTURE_LABELED = FORMAL_DIAG / "frozen_predictions_with_labels.csv"
WITHIN_INPUTS = (FORMAL_DIAG / "within_2018_2019/predictions.csv", FORMAL_DIAG / "within_2020_2021/predictions.csv")
RESOLVED = FORMAL_DIAG / "resolved_inputs.json"
REFERENCE_AUDIT = FORMAL_DIAG / "references/reference_audit.json"
MEMBERS = ("LR", "RF", "XGB", "SVM")
ARMS = MEMBERS + ("MEAN4",)
YEARS = (2018, 2019, 2020, 2021)
YEAR_COUNTS = {2018: 793, 2019: 399, 2020: 397, 2021: 400}
SEED = 20260904


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


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


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


def metric(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, int); probability = np.asarray(probability, float); pred = probability >= .5
    tn = int(np.sum((y == 0) & (~pred))); fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & (~pred))); tp = int(np.sum((y == 1) & pred))
    brier = float(np.mean((probability - y) ** 2))
    bins = np.minimum((np.clip(probability, 0, 1) * 10).astype(int), 9); ece = 0.0
    for value in range(10):
        selected = bins == value
        if np.any(selected):
            ece += float(np.mean(selected) * abs(np.mean(probability[selected]) - np.mean(y[selected])))
    return {"n": len(y), "BA": float((tp / max(tp + fn, 1) + tn / max(tn + fp, 1)) / 2),
        "AUC": binary_auc(y, probability), "F1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "accuracy": float(np.mean(pred == y)), "Brier": brier, "ECE10": ece,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp, "errors": fp + fn,
        "specificity": float(tn / max(tn + fp, 1)), "sensitivity": float(tp / max(tp + fn, 1))}


def mean_opposition(member_probability: np.ndarray, anchor_probability: np.ndarray) -> np.ndarray:
    member_probability = np.asarray(member_probability, float); decision = np.asarray(anchor_probability, float) >= .5
    return np.where(decision[:, None], 1 - member_probability, member_probability).mean(axis=1)


def grouped_top25(score: np.ndarray, sha: np.ndarray, groups: np.ndarray) -> np.ndarray:
    score = np.asarray(score, float); sha = np.asarray(sha).astype(str); groups = np.asarray(groups)
    route = np.zeros(len(score), bool)
    for group in sorted(set(groups.tolist())):
        indices = np.flatnonzero(groups == group); count = int(round(len(indices) * .25))
        order = np.lexsort((sha[indices], -score[indices])); route[indices[order[:count]]] = True
    return route


def route_metric(y: np.ndarray, probability: np.ndarray, member_probability: np.ndarray, sha: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    score = mean_opposition(member_probability, probability); route = grouped_top25(score, sha, groups)
    pred = probability >= .5; error = pred != y; fn = (y == 1) & (~pred); fp = (y == 0) & pred
    captured = error & route
    return {"routed": int(route.sum()), "captured_errors": int(captured.sum()),
        "error_recall25": float(captured.sum() / max(error.sum(), 1)), "route_precision": float(captured.sum() / max(route.sum(), 1)),
        "captured_FN": int(np.sum(fn & route)), "total_FN": int(fn.sum()), "FN_recall25": float(np.sum(fn & route) / max(fn.sum(), 1)),
        "captured_FP": int(np.sum(fp & route)), "total_FP": int(fp.sum()), "FP_recall25": float(np.sum(fp & route) / max(fp.sum(), 1))}


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False

