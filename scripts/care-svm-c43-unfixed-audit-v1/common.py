#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT = Path(os.environ.get("CSCUA_PROJECT", "/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT = Path(os.environ.get("CSCUA_OUTPUT", str(PROJECT / "outputs/care-svm-c43-unfixed-audit-v1"))).resolve()
SOURCE = PROJECT / "outputs/care-p31-classifier-family-stage2-v1/source_stage2_predictions.csv"
MAX_DELTA = 2.25
NEAR_ZERO = 0.05
SCALES = (0.0, .25, .5, .75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)


def norm(value: Any) -> str: return str(value or "").strip().casefold()


def value_of(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and str(row[name]).strip() != "": return row[name]
    raise RuntimeError("BLOCKED_BY_CONTRACT: missing column; expected one of " + repr(names))


def ensure_output(path: Path) -> Path:
    path = Path(path).resolve()
    try: path.relative_to(OUTPUT)
    except ValueError as exc: raise RuntimeError("BLOCKED_BY_CONTRACT: write outside experiment output") from exc
    path.parent.mkdir(parents=True, exist_ok=True); return path


def read_csv(path: Path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle: return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]):
    rows = list(rows); path = ensure_output(path)
    if not rows: path.write_text("", encoding="utf-8"); return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def dump_json(path: Path, value):
    path = ensure_output(path); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path): return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()


def asset_record(path: Path, purpose: str):
    path = Path(path).resolve(); stat = path.stat()
    return {"absolute_path": str(path), "purpose": purpose, "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": sha256_file(path)}


def logit(probability):
    value = np.clip(np.asarray(probability, float), 1e-12, 1 - 1e-12)
    return np.log(value / (1 - value))


def sigmoid(value):
    value = np.asarray(value, float)
    return np.where(value >= 0, 1 / (1 + np.exp(-value)), np.exp(value) / (1 + np.exp(value)))


def binary_auc(y, score):
    y = np.asarray(y, int); score = np.asarray(score, float); p = int(y.sum()); n = len(y) - p
    if p == 0 or n == 0: return float("nan")
    order = np.argsort(score, kind="mergesort"); ordered = score[order]; ranks = np.empty(len(y), float); start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and ordered[end] == ordered[start]: end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2; start = end
    return float((ranks[y == 1].sum() - p * (p + 1) / 2) / (p * n))


def metrics(y, base, final, route):
    y = np.asarray(y, int); base = np.asarray(base, float); final = np.asarray(final, float); route = np.asarray(route, bool)
    bp = base >= .5; pred = final >= .5; before = bp == y; after = pred == y; fixed = (~before) & after; broken = before & (~after)
    tn = int(np.sum((y == 0) & (~pred))); fp = int(np.sum((y == 0) & pred)); fn = int(np.sum((y == 1) & (~pred))); tp = int(np.sum((y == 1) & pred))
    return {"n": len(y), "BA": float((tn / max(tn + fp, 1) + tp / max(tp + fn, 1)) / 2),
        "AUC": binary_auc(y, final), "F1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp, "errors": fp + fn,
        "fixed": int(fixed.sum()), "broken": int(broken.sum()), "net": int(fixed.sum() - broken.sum()),
        "fixed_FN": int(np.sum(fixed & (y == 1))), "fixed_FP": int(np.sum(fixed & (y == 0))),
        "broken_TN": int(np.sum(broken & (y == 0))), "broken_TP": int(np.sum(broken & (y == 1))),
        "routed": int(route.sum()), "class_flips": int(np.sum(bp != pred))}


def load_source():
    rows = sorted([row for row in read_csv(SOURCE) if norm(row.get("arm")) == "svm"], key=lambda row: norm(row["sha256"]))
    if len(rows) != 799: raise RuntimeError("BLOCKED_BY_CONTRACT: SVM source row count changed")
    ids = [norm(row["sha256"]) for row in rows]
    y = np.asarray([int(float(value_of(row, "label", "y"))) for row in rows], int)
    fold = np.asarray([int(float(value_of(row, "fold", "outer_fold"))) for row in rows], int)
    base = np.asarray([float(value_of(row, "stage1_prob", "p_stage1")) for row in rows])
    final = np.asarray([float(value_of(row, "stage2_prob", "final_probability", "p_final")) for row in rows])
    route = np.asarray([int(float(value_of(row, "routed", "route"))) for row in rows], bool)
    return rows, ids, y, fold, base, final, route

