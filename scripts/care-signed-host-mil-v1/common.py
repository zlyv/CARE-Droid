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

PROJECT = Path(os.environ.get("CSHM_PROJECT", "/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT = Path(os.environ.get("CSHM_OUTPUT", str(PROJECT / "outputs/care-signed-host-mil-v1"))).resolve()
SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_STAGE2 = PROJECT / "outputs/care-p31-classifier-family-stage2-v1/source_stage2_predictions.csv"
BAG_CACHE = PROJECT / "artifacts/care-droid-formal-diagnostics-v1/cache/a0_role_2017_2021.npz"

SEED = 20261010
MAX_DELTA = 2.25
ARMS = ("M1_ABMIL", "M2_SIGNED_MIL", "M3_SIGNED_CLAM")
REAL_SEEDS = (SEED + 11, SEED + 29, SEED + 47)
NULL_REPETITIONS = int(os.environ.get("CSHM_NULL_REPETITIONS", "20"))
EPOCHS = int(os.environ.get("CSHM_EPOCHS", "25"))
BATCH_SIZE = 16


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
            if key not in fields: fields.append(key)
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
    if positive == 0 or negative == 0: return float("nan")
    order = np.argsort(score, kind="mergesort"); ordered = score[order]
    ranks = np.empty(len(score), float); start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and ordered[end] == ordered[start]: end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2; start = end
    return float((ranks[y == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def logit(probability):
    value = np.clip(np.asarray(probability, float), 1e-8, 1 - 1e-8)
    return np.log(value / (1 - value))


def sigmoid(value):
    value = np.asarray(value, float)
    return np.where(value >= 0, 1 / (1 + np.exp(-value)), np.exp(value) / (1 + np.exp(value)))


def correction_metrics(y, base, final, route) -> dict[str, Any]:
    y = np.asarray(y, int); base = np.asarray(base, float); final = np.asarray(final, float)
    route = np.asarray(route, bool); before_pred = base >= .5; pred = final >= .5
    before = before_pred == y; after = pred == y; fixed = (~before) & after; broken = before & (~after)
    tn = int(np.sum((y == 0) & (~pred))); fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & (~pred))); tp = int(np.sum((y == 1) & pred))
    return {"n": len(y), "BA": float((tn / max(tn + fp, 1) + tp / max(tp + fn, 1)) / 2),
            "AUC": binary_auc(y, final), "F1": float(2 * tp / max(2 * tp + fp + fn, 1)),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp, "errors": fp + fn,
            "fixed": int(fixed.sum()), "broken": int(broken.sum()), "net": int(fixed.sum() - broken.sum()),
            "fixed_FN": int(np.sum(fixed & (y == 1))), "fixed_FP": int(np.sum(fixed & (y == 0))),
            "broken_TN": int(np.sum(broken & (y == 0))), "broken_TP": int(np.sum(broken & (y == 1))),
            "routed": int(route.sum()), "class_flips": int(np.sum(before_pred != pred))}


def finite(value):
    return None if value is None or not np.isfinite(value) else float(value)


def spearman(x, y):
    from scipy.stats import spearmanr
    value = float(spearmanr(np.asarray(x, float), np.asarray(y, float)).statistic)
    return finite(value)


def load_source():
    rows = sorted([row for row in read_csv(SOURCE_STAGE2) if norm(row.get("arm")) == "svm"],
                  key=lambda row: norm(row["sha256"]))
    if len(rows) != 799 or len({norm(row["sha256"]) for row in rows}) != 799:
        raise RuntimeError("BLOCKED_BY_CONTRACT: SVM source rows changed")
    ids = [norm(row["sha256"]) for row in rows]
    y = np.asarray([int(float(row["label"])) for row in rows], int)
    fold = np.asarray([int(float(row["fold"])) for row in rows], int)
    base = np.asarray([float(row["stage1_prob"]) for row in rows], float)
    c43 = np.asarray([float(row["stage2_prob"]) for row in rows], float)
    route = np.asarray([int(float(row["routed"])) for row in rows], bool)
    with np.load(BAG_CACHE, allow_pickle=False) as archive:
        cache_ids = [norm(value) for value in archive["sha256"]]
        values = np.asarray(archive["host_values"], np.float32)
        offsets = np.asarray(archive["host_offsets"], int)
    if values.ndim != 2 or values.shape[1] != 169 or offsets.shape != (len(cache_ids) + 1,):
        raise RuntimeError("BLOCKED_BY_CONTRACT: formal Host cache shape changed")
    position = {sid: index for index, sid in enumerate(cache_ids)}; bags = []
    for sid in ids:
        if sid not in position: raise RuntimeError("BLOCKED_BY_CONTRACT: Host cache missing " + sid)
        index = position[sid]; bag = values[offsets[index]:offsets[index + 1], :43]
        if bag.ndim != 2 or bag.shape[0] < 1 or bag.shape[1] != 43:
            raise RuntimeError("BLOCKED_BY_CONTRACT: invalid C43 bag " + sid)
        bags.append(np.asarray(bag, np.float32))
    return rows, ids, y, fold, base, c43, route, bags

