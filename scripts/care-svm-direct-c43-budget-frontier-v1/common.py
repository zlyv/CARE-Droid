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

PROJECT = Path(os.environ.get("BRF_PROJECT", "/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT = Path(os.environ.get("BRF_OUTPUT", str(PROJECT / "outputs/care-svm-direct-c43-budget-frontier-v1"))).resolve()
DIRECT_METHOD = "P31SVM__SVMRoute__A0_ROLE_C43_RQ__DirectStage2"
BUDGETS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.75, 1.00)


def ensure_boundary(path: Path | None = None) -> Path:
    allowed = (PROJECT / "outputs").resolve()
    if OUTPUT == allowed or allowed not in OUTPUT.parents or not OUTPUT.name.startswith("care-svm-direct-c43-budget-frontier-v"):
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: unsafe output {OUTPUT}")
    target = OUTPUT if path is None else Path(path).resolve()
    if target != OUTPUT and OUTPUT not in target.parents:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: write outside output {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    rows = list(rows); path = ensure_boundary(path)
    if fields is None:
        fields = []
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
    path = ensure_boundary(path); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def asset(path: Path, purpose: str) -> dict[str, Any]:
    path = path.resolve(); stat = path.stat()
    return {"absolute_path": str(path), "purpose": purpose, "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": sha256_file(path)}


def discover() -> dict[str, Path]:
    candidates = {
        "predictions": [
            Path(os.environ["BRF_PREDICTIONS"]) if os.environ.get("BRF_PREDICTIONS") else None,
            PROJECT / "outputs/care-svm-c43-remove-bounded-correction-v1/source/U2_DIRECT_STAGE2/source_predictions.csv",
        ],
        "architecture": [
            Path(os.environ["BRF_ARCHITECTURE"]) if os.environ.get("BRF_ARCHITECTURE") else None,
            PROJECT / "outputs/care-svm-c43-remove-bounded-correction-v1/source/U2_DIRECT_STAGE2/architecture.json",
        ],
    }
    result = {}
    for name, paths in candidates.items():
        found = next((path.resolve() for path in paths if path is not None and path.is_file()), None)
        if found is None:
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: missing {name}; checked={[str(x) for x in paths if x is not None]}")
        result[name] = found
    return result


def norm(value: Any) -> str:
    return str(value or "").strip().casefold()


def route_at_budget(score, sha, fold, budget: float) -> np.ndarray:
    score = np.asarray(score, float); sha = np.asarray(sha).astype(str); fold = np.asarray(fold, int)
    route = np.zeros(len(score), bool)
    for current in sorted(set(fold.tolist())):
        index = np.flatnonzero(fold == current)
        count = int(round(len(index) * budget))
        order = np.lexsort((sha[index], -score[index]))
        route[index[order[:count]]] = True
    return route


def auc(y, score) -> float:
    y = np.asarray(y, int); score = np.asarray(score, float)
    positive = int(y.sum()); negative = len(y) - positive
    if not positive or not negative:
        return float("nan")
    order = np.argsort(score, kind="mergesort"); ordered = score[order]
    ranks = np.empty(len(y), float); start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and ordered[end] == ordered[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2; start = end
    return float((ranks[y == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def metrics(y, base, expert, route) -> dict[str, Any]:
    y = np.asarray(y, int); base = np.asarray(base, float); expert = np.asarray(expert, float); route = np.asarray(route, bool)
    final = np.where(route, expert, base)
    base_pred = base >= .5; pred = final >= .5
    before = base_pred == y; after = pred == y
    fixed = (~before) & after; broken = before & (~after); error = ~before
    tn = int(np.sum((y == 0) & (~pred))); fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & (~pred))); tp = int(np.sum((y == 1) & pred))
    captured = int(np.sum(route & error)); routed_correct = int(np.sum(route & before))
    return {
        "n": len(y), "BA": float((tn / max(tn + fp, 1) + tp / max(tp + fn, 1)) / 2),
        "AUC": auc(y, final), "F1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp, "errors": fp + fn,
        "fixed": int(fixed.sum()), "broken": int(broken.sum()), "net": int(fixed.sum() - broken.sum()),
        "fixed_FN": int(np.sum(fixed & (y == 1))), "fixed_FP": int(np.sum(fixed & (y == 0))),
        "broken_TN": int(np.sum(broken & (y == 0))), "broken_TP": int(np.sum(broken & (y == 1))),
        "routed": int(route.sum()), "routed_stage1_errors": captured, "routed_stage1_correct": routed_correct,
        "error_recall": float(captured / max(error.sum(), 1)), "route_precision": float(captured / max(route.sum(), 1)),
        "expert_fix_rate": float(fixed.sum() / max(captured, 1)),
        "expert_damage_rate": float(broken.sum() / max(routed_correct, 1)),
        "class_flips": int(np.sum(base_pred != pred)),
    }


def mcnemar_exact(candidate_ok, baseline_ok) -> tuple[int, int, float]:
    candidate_ok = np.asarray(candidate_ok, bool); baseline_ok = np.asarray(baseline_ok, bool)
    a = int(np.sum(candidate_ok & (~baseline_ok))); b = int(np.sum((~candidate_ok) & baseline_ok)); n = a + b
    if n == 0:
        return a, b, 1.0
    tail = sum(math.comb(n, k) for k in range(min(a, b) + 1)) / (2 ** n)
    return a, b, float(min(1.0, 2 * tail))
