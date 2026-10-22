#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(os.environ.get("MALEVAL_PROJECT", "/home/boyuanjie/projects/tasa-droid")).resolve()
DATASET = Path(os.environ.get(
    "MALEVAL_DATASET",
    "/home/boyuanjie/datasets/android/external_benchmarks/MalEval2026",
)).resolve()
OUTPUT = Path(os.environ.get(
    "MALEVAL_OUTPUT",
    str(PROJECT / "outputs/care-maleval2026-direct-route-budget-v1"),
)).resolve()
SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS = PROJECT / "scripts"

SOURCE_P31 = PROJECT / "artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz"
FORMAL_DIAG = PROJECT / "artifacts/care-droid-formal-diagnostics-v1"
TRAIN_WINDOW = PROJECT / "outputs/care-svm-train-window-expansion-v1"
SOURCE_SVM = PROJECT / "outputs/care-p31-classifier-family-stage2-v1/source/SVM/source_predictions.csv"
SOURCE_RESOLVED = PROJECT / "artifacts/care-litelocal-semantic-source-v1/resolved_inputs.json"

COHORTS = {"benign": (25, 0), "malradar": (200, 1), "new": (30, 1)}
EXPECTED_N = 255
MAX_DELTA = 2.25
SEED = 20261010 + 1718
MEMBERS = ("LR", "RF", "XGB", "SVM")
BUDGETS = (25, 30, 40, 50)


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


def dump_json(path: Path, value: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def metrics(y, probability) -> dict[str, Any]:
    y = np.asarray(y, int); probability = np.asarray(probability, float); pred = probability >= .5
    tn = int(np.sum((y == 0) & (~pred))); fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & (~pred))); tp = int(np.sum((y == 1) & pred))
    has_negative = (tn + fp) > 0; has_positive = (tp + fn) > 0
    specificity = tn / (tn + fp) if has_negative else float("nan")
    sensitivity = tp / (tp + fn) if has_positive else float("nan")
    balanced_accuracy = (specificity + sensitivity) / 2 if has_negative and has_positive else float("nan")
    return {"n": len(y), "BA": float(balanced_accuracy),
            "AUC": binary_auc(y, probability), "F1": float(2 * tp / max(2 * tp + fp + fn, 1)),
            "accuracy": float(np.mean(pred == y)), "specificity": float(specificity),
            "sensitivity": float(sensitivity), "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "errors": fp + fn}


def mean_opposition(member_probability, anchor_probability):
    member_probability = np.asarray(member_probability, float)
    decision = np.asarray(anchor_probability, float) >= .5
    return np.where(decision[:, None], 1 - member_probability, member_probability).mean(axis=1)


def arm_probability(member_probability, arm: str):
    member_probability = np.asarray(member_probability, float)
    if arm == "MEAN4": return member_probability.mean(axis=1)
    return member_probability[:, MEMBERS.index(arm)]


def top25(score, sha):
    return top_fraction(score, sha, .25)


def top_fraction(score, sha, fraction):
    score = np.asarray(score, float); sha = np.asarray(sha).astype(str)
    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("route fraction must be in (0,1]")
    count = int(round(len(score) * fraction)); route = np.zeros(len(score), bool)
    order = np.lexsort((sha, -score)); route[order[:count]] = True
    return route

