#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT = Path(os.environ.get("SVMPBE_PROJECT", "/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT = Path(os.environ.get(
    "SVMPBE_OUTPUT", str(PROJECT / "outputs/care-svm-positive-benign-evidence-v1")
)).resolve()

SOURCE_STAGE2 = PROJECT / "outputs/care-p31-classifier-family-stage2-v1/source_stage2_predictions.csv"
FROZEN_STAGE2 = PROJECT / "outputs/care-p31-classifier-family-stage2-v1/frozen_future_stage2_predictions.csv"
SOURCE_P31 = PROJECT / "artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz"
FUNCTION_INVENTORY = PROJECT / "outputs/care-function-role-explanation-v1/function_feature_inventory.csv"
FORMAL_DIAG = PROJECT / "artifacts/care-droid-formal-diagnostics-v1"

SEED = 20261009
MAX_DELTA = 2.25
YEARS = (2018, 2019, 2020, 2021)

# Deliberately excludes families that directly duplicate sensitive Role identity.
# A positive certificate must come from observed non-sensitive functional context.
CONTEXT_FAMILIES = (
    "UI_INTERACTION",
    "WEB_CONTENT",
    "MEDIA_PRESENTATION",
    "NOTIFICATION",
    "DOCUMENT_WORKFLOW",
    "BACKGROUND_SCHEDULING",
    "IPC_COMPONENT",
)


def norm(value: Any) -> str:
    return str(value or "").strip().casefold()


def ensure_output(path: Path) -> Path:
    path = Path(path).resolve()
    try:
        path.relative_to(OUTPUT)
    except ValueError as exc:
        raise RuntimeError("BLOCKED_BY_CONTRACT: write outside experiment output: " + str(path)) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows); path = ensure_output(path)
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
    path = ensure_output(path); temporary = path.with_suffix(path.suffix + ".tmp")
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


def asset_record(path: Path, purpose: str) -> dict[str, Any]:
    path = Path(path).resolve(); stat = path.stat()
    return {"absolute_path": str(path), "purpose": purpose, "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": sha256_file(path)}


def binary_auc(y, score) -> float:
    y = np.asarray(y, int); score = np.asarray(score, float)
    positive = int(y.sum()); negative = len(y) - positive
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort"); ordered = score[order]
    rank = np.empty(len(score), float); start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and ordered[end] == ordered[start]:
            end += 1
        rank[order[start:end]] = (start + 1 + end) / 2; start = end
    return float((rank[y == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def logit(probability):
    value = np.clip(np.asarray(probability, float), 1e-8, 1 - 1e-8)
    return np.log(value / (1 - value))


def sigmoid(value):
    value = np.asarray(value, float)
    return np.where(value >= 0, 1 / (1 + np.exp(-value)), np.exp(value) / (1 + np.exp(value)))


def metrics(y, base, final, route) -> dict[str, Any]:
    y = np.asarray(y, int); base = np.asarray(base, float); final = np.asarray(final, float)
    route = np.asarray(route, bool); base_pred = base >= .5; pred = final >= .5
    before = base_pred == y; after = pred == y
    tn = int(np.sum((y == 0) & (~pred))); fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & (~pred))); tp = int(np.sum((y == 1) & pred))
    fixed = (~before) & after; broken = before & (~after)
    return {"n": len(y), "BA": float((tn / max(tn + fp, 1) + tp / max(tp + fn, 1)) / 2),
            "AUC": binary_auc(y, final), "F1": float(2 * tp / max(2 * tp + fp + fn, 1)),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp, "errors": fp + fn,
            "fixed": int(fixed.sum()), "broken": int(broken.sum()), "net": int(fixed.sum() - broken.sum()),
            "fixed_FN": int(np.sum(fixed & (y == 1))), "fixed_FP": int(np.sum(fixed & (y == 0))),
            "broken_TN": int(np.sum(broken & (y == 0))), "broken_TP": int(np.sum(broken & (y == 1))),
            "routed": int(route.sum()), "class_flips": int(np.sum(base_pred != pred))}


def spearman(x, y):
    from scipy.stats import spearmanr
    value = float(spearmanr(np.asarray(x, float), np.asarray(y, float)).statistic)
    return None if not math.isfinite(value) else value

