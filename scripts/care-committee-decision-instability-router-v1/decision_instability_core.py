#!/usr/bin/env python3
"""Decision-instability routing helpers for frozen committee probabilities."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


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


def decision_instability(
    stage1_probability: np.ndarray,
    member_probability: np.ndarray,
) -> dict[str, np.ndarray]:
    """Measure committee opposition to the frozen Stage-I hard decision.

    For a Stage-I malware decision, support is mean(p_m); for a benign
    decision, support is mean(1-p_m).  U_DI = 1-support.  Algebraically this is
    the committee mean probability assigned to the class opposite Stage I.
    """
    base = np.asarray(stage1_probability, dtype=np.float64)
    member = np.asarray(member_probability, dtype=np.float64)
    if base.ndim != 1 or member.shape != (len(base), 4):
        raise ValueError("expected Stage-I [n] and committee [n,4]")
    if not np.isfinite(base).all() or not np.isfinite(member).all():
        raise ValueError("probabilities must be finite")
    if np.any((base < 0.0) | (base > 1.0)) or np.any(
        (member < 0.0) | (member > 1.0)
    ):
        raise ValueError("probabilities must be in [0,1]")
    decision = (base >= 0.5).astype(np.float64)
    support_by_member = (
        decision[:, None] * member
        + (1.0 - decision[:, None]) * (1.0 - member)
    )
    support = np.mean(support_by_member, axis=1)
    uncertainty = 1.0 - support
    mean_probability = np.mean(member, axis=1)
    simplified = np.where(decision == 1.0, 1.0 - mean_probability, mean_probability)
    if not np.allclose(uncertainty, simplified, atol=1e-12, rtol=0.0):
        raise RuntimeError("decision-instability algebraic audit failed")
    return {
        "stage1_decision": decision.astype(np.int64),
        "support_by_member": support_by_member,
        "committee_support": support,
        "uncertainty": uncertainty,
        "p_bar": mean_probability,
    }


def exact_top_fraction(score: np.ndarray, budget: float) -> np.ndarray:
    value = np.asarray(score, dtype=np.float64)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError("score must be finite and one-dimensional")
    if not 0.0 < float(budget) < 1.0:
        raise ValueError("budget must be in (0,1)")
    count = max(1, min(len(value), int(round(float(budget) * len(value)))))
    order = np.argsort(value, kind="mergesort")
    route = np.zeros(len(value), dtype=bool)
    route[order[-count:]] = True
    return route


def groupwise_top_fraction(
    score: np.ndarray,
    groups: np.ndarray,
    budget: float,
) -> np.ndarray:
    value = np.asarray(score, dtype=np.float64)
    group = np.asarray(groups)
    if value.shape != group.shape:
        raise ValueError("score and group arrays must align")
    route = np.zeros(len(value), dtype=bool)
    for item in np.unique(group):
        index = np.flatnonzero(group == item)
        route[index] = exact_top_fraction(value[index], budget)
    return route


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def evaluate_route(
    method: str,
    labels: np.ndarray,
    stage1_probability: np.ndarray,
    expert_probability: np.ndarray,
    route: np.ndarray,
    score: np.ndarray,
    *,
    year: int | None = None,
    high_confidence_threshold: float = 0.8,
) -> dict:
    y = np.asarray(labels, dtype=np.int64)
    base = np.asarray(stage1_probability, dtype=np.float64)
    expert = np.asarray(expert_probability, dtype=np.float64)
    selected = np.asarray(route, dtype=bool)
    value = np.asarray(score, dtype=np.float64)
    if not (y.shape == base.shape == expert.shape == selected.shape == value.shape):
        raise ValueError("route arrays are not aligned")
    base_correct = (base >= 0.5) == y
    expert_correct = (expert >= 0.5) == y
    error = ~base_correct
    repairable = error & expert_correct
    unrepairable = error & ~expert_correct
    harmful = base_correct & ~expert_correct
    confidence = np.maximum(base, 1.0 - base)
    high_error = error & (confidence >= float(high_confidence_threshold))
    accepted = ~selected
    final_probability = np.where(selected, expert, base)
    final_prediction = final_probability >= 0.5
    fixed = int(np.sum(selected & repairable))
    broken = int(np.sum(selected & harmful))
    captured_error = int(np.sum(selected & error))
    result = {
        "method": method,
        "n": int(len(y)),
        "routed_n": int(np.sum(selected)),
        "routed_ratio": float(np.mean(selected)),
        "base_error_n": int(np.sum(error)),
        "routed_base_error_n": captured_error,
        "error_recall": safe_ratio(captured_error, int(np.sum(error))),
        "high_confidence_error_n": int(np.sum(high_error)),
        "routed_high_confidence_error_n": int(np.sum(selected & high_error)),
        "high_confidence_error_recall": safe_ratio(
            int(np.sum(selected & high_error)), int(np.sum(high_error))
        ),
        "routing_precision": safe_ratio(captured_error, int(np.sum(selected))),
        "accepted_risk": float(np.mean(error[accepted])) if np.any(accepted) else 0.0,
        "repairable_error_n": int(np.sum(repairable)),
        "routed_repairable_error_n": fixed,
        "repairable_error_recall_at_budget": safe_ratio(
            fixed, int(np.sum(repairable))
        ),
        "harmful_sample_n": int(np.sum(harmful)),
        "routed_harmful_sample_n": broken,
        "harmful_capture": safe_ratio(broken, int(np.sum(harmful))),
        "unrepairable_error_n": int(np.sum(unrepairable)),
        "routed_unrepairable_error_n": int(np.sum(selected & unrepairable)),
        "unrepairable_capture": safe_ratio(
            int(np.sum(selected & unrepairable)), int(np.sum(unrepairable))
        ),
        "BA": float(balanced_accuracy_score(y, final_prediction)),
        "AUC": float(roc_auc_score(y, final_probability)),
        "F1": float(f1_score(y, final_prediction, zero_division=0)),
        "fixed": fixed,
        "broken": broken,
        "net": fixed - broken,
        "score_mean": float(np.mean(value)),
        "score_max": float(np.max(value)),
    }
    if year is not None:
        result["year"] = int(year)
        result["diagnostic_only"] = True
    return result


def baseline_metrics(
    labels: np.ndarray,
    stage1_probability: np.ndarray,
    *,
    year: int | None = None,
) -> dict:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(stage1_probability, dtype=np.float64)
    prediction = p >= 0.5
    result = {
        "method": "Global24",
        "n": int(len(y)),
        "routed_n": 0,
        "routed_ratio": 0.0,
        "BA": float(balanced_accuracy_score(y, prediction)),
        "AUC": float(roc_auc_score(y, p)),
        "F1": float(f1_score(y, prediction, zero_division=0)),
        "fixed": 0,
        "broken": 0,
        "net": 0,
    }
    if year is not None:
        result["year"] = int(year)
        result["diagnostic_only"] = True
    return result


def overlap_row(
    left_name: str,
    left_route: np.ndarray,
    right_name: str,
    right_route: np.ndarray,
    *,
    scope: str,
) -> dict:
    left = np.asarray(left_route, dtype=bool)
    right = np.asarray(right_route, dtype=bool)
    intersection = int(np.sum(left & right))
    union = int(np.sum(left | right))
    return {
        "scope": scope,
        "router_a": left_name,
        "router_b": right_name,
        "router_a_n": int(np.sum(left)),
        "router_b_n": int(np.sum(right)),
        "intersection_n": intersection,
        "union_n": union,
        "jaccard": safe_ratio(intersection, union),
        "a_covered_by_b": safe_ratio(intersection, int(np.sum(left))),
        "b_covered_by_a": safe_ratio(intersection, int(np.sum(right))),
    }
