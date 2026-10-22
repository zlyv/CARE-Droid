#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

PROJECT = Path("/home/boyuanjie/projects/tasa-droid")
OUTPUT = PROJECT / "outputs/care-group-calibration-compact43-validation-v1"
FORMAL = PROJECT / "artifacts/care-p31rf-c43-haig-host-ablation-source-v1/arms/A0_ROLE_ONLY/source_predictions.csv"
P31 = PROJECT / "artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz"
HOST = PROJECT / "artifacts/care-p31rf-c43-haig-host-ablation-source-v1/cache/A0_ROLE_ONLY.npz"
METHOD = "P31RF__RFRoute__A0_ROLE_ONLY__Bounded"
SEED = 20261008


def norm(value):
    return str(value).strip().casefold()


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def auc(y, score):
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    pos = int(y.sum())
    neg = len(y) - pos
    if not pos or not neg:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    sorted_score = score[order]
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_score[end] == sorted_score[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def metrics(y, probability):
    from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score
    y = np.asarray(y, int)
    probability = np.asarray(probability, float)
    pred = probability >= .5
    return {
        "n": int(len(y)), "BA": float(balanced_accuracy_score(y, pred)),
        "AUC": auc(y, probability), "F1": float(f1_score(y, pred, zero_division=0)),
        "Brier": float(brier_score_loss(y, probability)), "errors": int(np.sum(pred != y)),
        "TN": int(np.sum((y == 0) & ~pred)), "FP": int(np.sum((y == 0) & pred)),
        "FN": int(np.sum((y == 1) & ~pred)), "TP": int(np.sum((y == 1) & pred)),
    }

