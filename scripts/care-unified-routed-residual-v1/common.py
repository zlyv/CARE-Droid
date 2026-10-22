#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


PROJECT = Path(os.environ.get("CURR_PROJECT", "/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT = Path(os.environ.get("CURR_OUTPUT", str(PROJECT / "outputs/care-unified-routed-residual-v1"))).resolve()
MEMBER = PROJECT / "artifacts/care-litelocal-stage1-attribution-source-v1/source_member_oof_predictions.csv"
FORMAL = PROJECT / "artifacts/care-p31rf-c43-haig-host-ablation-source-v1/arms/A0_ROLE_ONLY/source_predictions.csv"
MAX_DELTA = 2.25
FORMAL_METHOD = "P31RF__RFRoute__A0_ROLE_ONLY__Bounded"


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


def asset_record(path: Path, purpose: str) -> dict[str, Any]:
    path = Path(path).resolve(); stat = path.stat()
    return {"path": str(path), "purpose": purpose, "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": sha256_file(path)}


def logit(probability):
    probability = np.clip(np.asarray(probability, float), 1e-8, 1 - 1e-8)
    return np.log(probability / (1 - probability))


def sigmoid(value):
    value = np.asarray(value, float)
    return np.where(value >= 0, 1 / (1 + np.exp(-value)), np.exp(value) / (1 + np.exp(value)))


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


def classification_metrics(y, probability) -> dict[str, Any]:
    y = np.asarray(y, int); probability = np.asarray(probability, float); pred = probability >= .5
    tn = int(np.sum((y == 0) & (~pred))); fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & (~pred))); tp = int(np.sum((y == 1) & pred))
    return {"n": len(y), "BA": float((tn / max(tn + fp, 1) + tp / max(tp + fn, 1)) / 2),
            "AUC": binary_auc(y, probability), "F1": float(2 * tp / max(2 * tp + fp + fn, 1)),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp, "errors": fp + fn}


def correction_metrics(y, base, final, route) -> dict[str, Any]:
    y = np.asarray(y, int); base = np.asarray(base, float); final = np.asarray(final, float)
    route = np.asarray(route, bool); bp = base >= .5; fpred = final >= .5
    base_ok = bp == y; final_ok = fpred == y
    fixed = (~base_ok) & final_ok; broken = base_ok & (~final_ok)
    return {**classification_metrics(y, final), "fixed": int(fixed.sum()), "broken": int(broken.sum()),
            "net": int(fixed.sum() - broken.sum()), "fixed_FN": int(np.sum(fixed & (y == 1))),
            "fixed_FP": int(np.sum(fixed & (y == 0))), "broken_TN": int(np.sum(broken & (y == 0))),
            "broken_TP": int(np.sum(broken & (y == 1))), "routed": int(route.sum()),
            "routed_errors": int(np.sum(route & (~base_ok))), "routed_correct": int(np.sum(route & base_ok)),
            "class_flips": int(np.sum(bp != fpred))}
