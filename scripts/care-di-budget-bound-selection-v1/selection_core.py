#!/usr/bin/env python3
"""Shared helpers for source-only DI budget/bound selection."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


MEMBER_COLUMNS = ("p_lr", "p_rf", "p_xgb", "p_svm")


def read_csv(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def norm_sha(value) -> str:
    return str(value or "").strip().casefold()


def array(rows: list[dict], name: str, dtype=float) -> np.ndarray:
    return np.asarray([dtype(row[name]) for row in rows])


def parse_float_list(text: str) -> list[float]:
    result: list[float] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if value not in result:
            result.append(value)
    if not result:
        raise ValueError("float list is empty")
    return result


def cap_tag(value: float) -> str:
    text = ("%.6f" % float(value)).rstrip("0").rstrip(".")
    return "max_delta_" + text.replace("-", "m").replace(".", "p")


def decision_instability(base: np.ndarray, members: np.ndarray) -> np.ndarray:
    base = np.asarray(base, dtype=np.float64)
    members = np.asarray(members, dtype=np.float64)
    if members.shape != (len(base), 4):
        raise ValueError("expected four committee probabilities per sample")
    decision = (base >= 0.5).astype(np.float64)
    support = decision[:, None] * members + (1.0 - decision[:, None]) * (1.0 - members)
    return 1.0 - np.mean(support, axis=1)


def exact_top_fraction(score: np.ndarray, budget: float) -> np.ndarray:
    value = np.asarray(score, dtype=np.float64)
    count = max(1, min(len(value), int(round(float(budget) * len(value)))))
    order = np.argsort(value, kind="mergesort")
    route = np.zeros(len(value), dtype=bool)
    route[order[-count:]] = True
    return route


def groupwise_top_fraction(score: np.ndarray, group: np.ndarray, budget: float) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64)
    group = np.asarray(group)
    route = np.zeros(len(score), dtype=bool)
    for item in np.unique(group):
        index = np.flatnonzero(group == item)
        route[index] = exact_top_fraction(score[index], budget)
    return route


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def evaluate(
    labels: np.ndarray,
    base: np.ndarray,
    expert: np.ndarray,
    route: np.ndarray,
) -> dict:
    y = np.asarray(labels, dtype=np.int64)
    base = np.asarray(base, dtype=np.float64)
    expert = np.asarray(expert, dtype=np.float64)
    route = np.asarray(route, dtype=bool)
    if not (y.shape == base.shape == expert.shape == route.shape):
        raise ValueError("evaluation arrays do not align")
    base_pred = base >= 0.5
    expert_pred = expert >= 0.5
    base_correct = base_pred == y
    expert_correct = expert_pred == y
    error = ~base_correct
    repairable = error & expert_correct
    harmful = base_correct & ~expert_correct
    unrepairable = error & ~expert_correct
    confidence = np.maximum(base, 1.0 - base)
    high_error = error & (confidence >= 0.8)
    final = np.where(route, expert, base)
    pred = final >= 0.5
    fixed = int(np.sum(route & repairable))
    broken = int(np.sum(route & harmful))
    routed = int(np.sum(route))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tp = int(np.sum((pred == 1) & (y == 1)))
    accepted = ~route
    return {
        "n": int(len(y)),
        "routed_n": routed,
        "routed_ratio": float(np.mean(route)),
        "compute_saving_ratio": float(1.0 - np.mean(route)),
        "BA": float(balanced_accuracy_score(y, pred)),
        "AUC": float(roc_auc_score(y, final)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": safe_ratio(tn, tn + fp),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "base_error_n": int(np.sum(error)),
        "routed_base_error_n": int(np.sum(route & error)),
        "error_recall": safe_ratio(int(np.sum(route & error)), int(np.sum(error))),
        "high_confidence_error_n": int(np.sum(high_error)),
        "routed_high_confidence_error_n": int(np.sum(route & high_error)),
        "high_confidence_error_recall": safe_ratio(
            int(np.sum(route & high_error)), int(np.sum(high_error))
        ),
        "routing_precision": safe_ratio(int(np.sum(route & error)), routed),
        "accepted_risk": float(np.mean(error[accepted])) if np.any(accepted) else 0.0,
        "repairable_error_n": int(np.sum(repairable)),
        "routed_repairable_error_n": fixed,
        "repairable_error_recall": safe_ratio(fixed, int(np.sum(repairable))),
        "harmful_sample_n": int(np.sum(harmful)),
        "routed_harmful_sample_n": broken,
        "harmful_capture": safe_ratio(broken, int(np.sum(harmful))),
        "unrepairable_error_n": int(np.sum(unrepairable)),
        "routed_unrepairable_error_n": int(np.sum(route & unrepairable)),
        "unrepairable_capture": safe_ratio(
            int(np.sum(route & unrepairable)), int(np.sum(unrepairable))
        ),
        "fixed": fixed,
        "broken": broken,
        "net": fixed - broken,
        "fixed_per_routed": safe_ratio(fixed, routed),
        "broken_per_routed": safe_ratio(broken, routed),
        "net_per_routed": safe_ratio(fixed - broken, routed),
    }


def baseline(labels: np.ndarray, probability: np.ndarray) -> dict:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    pred = p >= 0.5
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tp = int(np.sum((pred == 1) & (y == 1)))
    return {
        "n": int(len(y)),
        "BA": float(balanced_accuracy_score(y, pred)),
        "AUC": float(roc_auc_score(y, p)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": safe_ratio(tn, tn + fp),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def find_column(rows: list[dict], preferred: str, contains: tuple[str, ...]) -> str:
    if not rows:
        raise RuntimeError("prediction file is empty")
    names = list(rows[0])
    if preferred in names:
        return preferred
    candidates = [name for name in names if all(token in name for token in contains)]
    if len(candidates) != 1:
        raise RuntimeError(
            "could not identify column; preferred=%r contains=%r candidates=%r"
            % (preferred, contains, candidates)
        )
    return candidates[0]


def route_threshold(score: np.ndarray, route: np.ndarray) -> float:
    score = np.asarray(score, dtype=np.float64)
    route = np.asarray(route, dtype=bool)
    if not np.any(route):
        return math.inf
    return float(np.min(score[route]))

