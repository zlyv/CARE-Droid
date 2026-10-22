#!/usr/bin/env python3
"""Strict calibrated diverse-classifier committee uncertainty helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Callable

import numpy as np


MEMBER_NAMES = ("LR", "RF", "XGB", "SVM")


def binary_entropy(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / math.log(2.0)


def committee_scores(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    value = np.asarray(probabilities, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != len(MEMBER_NAMES):
        raise ValueError("committee probabilities must have shape [n,4]")
    if not np.isfinite(value).all() or np.any((value < 0.0) | (value > 1.0)):
        raise ValueError("committee probabilities must be finite in [0,1]")
    mean_probability = np.mean(value, axis=1)
    predictive_entropy = binary_entropy(mean_probability)
    expected_entropy = np.mean(binary_entropy(value), axis=1)
    mutual_information = np.maximum(predictive_entropy - expected_entropy, 0.0)
    return {
        "p_bar": mean_probability,
        "predictive_entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": mutual_information,
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
    score: np.ndarray, groups: np.ndarray, budget: float
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


def clipped_logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    return np.log(p / (1.0 - p))


def make_member(name: str, seed: int, jobs: int):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if name == "LR":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="liblinear",
                max_iter=5000,
                random_state=int(seed),
            )),
        ])
    if name == "RF":
        return RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=int(jobs),
            random_state=int(seed),
        )
    if name == "XGB":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "xgboost is required by the fixed four-member committee; "
                "use the existing MalScan-capable environment"
            ) from exc
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=int(jobs),
            random_state=int(seed),
            verbosity=0,
        )
    if name == "SVM":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", SVC(
                C=1.0,
                kernel="rbf",
                gamma="scale",
                probability=False,
                random_state=int(seed),
            )),
        ])
    raise ValueError("unknown committee member: " + str(name))


def raw_score(model, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        value = np.asarray(model.decision_function(features), dtype=np.float64)
        return value.reshape(-1)
    if hasattr(model, "predict_proba"):
        probability = np.asarray(model.predict_proba(features), dtype=np.float64)
        classes = [int(value) for value in model.classes_]
        if 1 not in classes:
            raise RuntimeError("classifier has no positive class")
        return clipped_logit(probability[:, classes.index(1)])
    raise RuntimeError("committee member exposes no continuous decision score")


def fit_platt(raw_oof: np.ndarray, labels: np.ndarray, seed: int):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=2000,
        random_state=int(seed),
    )
    model.fit(np.asarray(raw_oof).reshape(-1, 1), labels)
    return model


@dataclass
class CommitteeFit:
    target_probability: np.ndarray
    inner_oof_probability: np.ndarray
    audit: list[dict]


def fit_calibrated_committee(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_target: np.ndarray,
    *,
    inner_folds: int,
    split_seed: int,
    model_seed: int,
    jobs: int,
) -> CommitteeFit:
    """Fit models/calibrators on train only, then predict an unlabeled target."""
    from sklearn.model_selection import StratifiedKFold

    x_fit = np.asarray(x_train, dtype=np.float64)
    y_fit = np.asarray(y_train, dtype=np.int64)
    x_eval = np.asarray(x_target, dtype=np.float64)
    if x_fit.ndim != 2 or x_eval.ndim != 2 or x_fit.shape[1] != x_eval.shape[1]:
        raise ValueError("train and target Global24 matrices are incompatible")
    if x_fit.shape[1] != 24:
        raise RuntimeError("committee must receive exactly Global24")
    if y_fit.shape != (len(x_fit),) or not np.isin(y_fit, (0, 1)).all():
        raise ValueError("invalid training labels")

    splitter = StratifiedKFold(
        n_splits=int(inner_folds), shuffle=True, random_state=int(split_seed)
    )
    splits = list(splitter.split(np.zeros(len(y_fit)), y_fit))
    target_probability = np.zeros((len(x_eval), len(MEMBER_NAMES)), dtype=np.float64)
    oof_probability = np.zeros((len(x_fit), len(MEMBER_NAMES)), dtype=np.float64)
    audits: list[dict] = []
    for column, name in enumerate(MEMBER_NAMES):
        raw_oof = np.full(len(x_fit), np.nan, dtype=np.float64)
        for inner_fold, (proper_index, calibration_index) in enumerate(splits, start=1):
            model = make_member(
                name,
                model_seed + 10000 * column + 100 * inner_fold,
                jobs,
            )
            model.fit(x_fit[proper_index], y_fit[proper_index])
            raw_oof[calibration_index] = raw_score(model, x_fit[calibration_index])
        if not np.isfinite(raw_oof).all():
            raise RuntimeError("incomplete inner OOF raw score for " + name)
        calibrator = fit_platt(
            raw_oof, y_fit, model_seed + 10000 * column + 999
        )
        oof_probability[:, column] = np.clip(
            calibrator.predict_proba(raw_oof.reshape(-1, 1))[:, 1],
            1e-8,
            1.0 - 1e-8,
        )
        final_model = make_member(name, model_seed + 10000 * column, jobs)
        final_model.fit(x_fit, y_fit)
        target_raw = raw_score(final_model, x_eval)
        target_probability[:, column] = np.clip(
            calibrator.predict_proba(target_raw.reshape(-1, 1))[:, 1],
            1e-8,
            1.0 - 1e-8,
        )
        audits.append({
            "member": name,
            "train_n": int(len(x_fit)),
            "target_n": int(len(x_eval)),
            "dimension": int(x_fit.shape[1]),
            "inner_folds": int(inner_folds),
            "calibration": "Platt sigmoid fitted on train-only inner OOF raw scores",
            "target_labels_received": False,
            "raw_oof_min": float(np.min(raw_oof)),
            "raw_oof_max": float(np.max(raw_oof)),
            "calibrated_oof_mean": float(np.mean(oof_probability[:, column])),
        })
    return CommitteeFit(target_probability, oof_probability, audits)


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def routing_metrics(
    method: str,
    labels: np.ndarray,
    stage1_probability: np.ndarray,
    expert_probability: np.ndarray,
    route: np.ndarray,
    score: np.ndarray,
    *,
    high_confidence_threshold: float = 0.8,
) -> dict:
    y = np.asarray(labels, dtype=np.int64)
    base = np.asarray(stage1_probability, dtype=np.float64)
    expert = np.asarray(expert_probability, dtype=np.float64)
    selected = np.asarray(route, dtype=bool)
    risk_score = np.asarray(score, dtype=np.float64)
    if not (y.shape == base.shape == expert.shape == selected.shape == risk_score.shape):
        raise ValueError("routing metric inputs are not aligned")
    base_correct = (base >= 0.5) == y
    expert_correct = (expert >= 0.5) == y
    error = ~base_correct
    repairable = error & expert_correct
    unrepairable = error & ~expert_correct
    harmful = base_correct & ~expert_correct
    confidence = np.maximum(base, 1.0 - base)
    high_error = error & (confidence >= float(high_confidence_threshold))
    accepted = ~selected
    fixed = int(np.sum(selected & repairable))
    broken = int(np.sum(selected & harmful))
    captured_error = int(np.sum(selected & error))
    return {
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
        "accepted_n": int(np.sum(accepted)),
        "accepted_error_n": int(np.sum(accepted & error)),
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
        "fixed": fixed,
        "broken": broken,
        "net": fixed - broken,
        "score_mean": float(np.mean(risk_score)),
        "score_max": float(np.max(risk_score)),
    }


def expected_calibration_error(
    labels: np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    total = 0.0
    for index in range(int(bins)):
        if index + 1 == int(bins):
            selected = (p >= edges[index]) & (p <= edges[index + 1])
        else:
            selected = (p >= edges[index]) & (p < edges[index + 1])
        if np.any(selected):
            total += float(np.mean(selected)) * abs(
                float(np.mean(p[selected])) - float(np.mean(y[selected]))
            )
    return float(total)


def member_metric_rows(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    scope: str,
    fold: int | None = None,
) -> list[dict]:
    from sklearn.metrics import (
        balanced_accuracy_score,
        brier_score_loss,
        f1_score,
        roc_auc_score,
    )

    y = np.asarray(labels, dtype=np.int64)
    value = np.asarray(probabilities, dtype=np.float64)
    rows = []
    for column, name in enumerate(MEMBER_NAMES):
        p = value[:, column]
        prediction = (p >= 0.5).astype(np.int64)
        row = {
            "scope": scope,
            "member": name,
            "n": int(len(y)),
            "BA": float(balanced_accuracy_score(y, prediction)),
            "AUC": float(roc_auc_score(y, p)),
            "F1": float(f1_score(y, prediction, zero_division=0)),
            "Brier": float(brier_score_loss(y, p)),
            "ECE10": expected_calibration_error(y, p, 10),
        }
        if fold is not None:
            row["fold"] = int(fold)
        rows.append(row)
    return rows


def average_ranks(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(len(x), dtype=np.float64)
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and sorted_x[end] == sorted_x[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def diversity_rows(
    probabilities: np.ndarray,
    *,
    scope: str,
    fold: int | None = None,
) -> list[dict]:
    value = np.asarray(probabilities, dtype=np.float64)
    rows = []
    for left, right in combinations(range(len(MEMBER_NAMES)), 2):
        p_left, p_right = value[:, left], value[:, right]
        pearson = float(np.corrcoef(p_left, p_right)[0, 1])
        spearman = float(np.corrcoef(
            average_ranks(p_left), average_ranks(p_right)
        )[0, 1])
        row = {
            "scope": scope,
            "member_a": MEMBER_NAMES[left],
            "member_b": MEMBER_NAMES[right],
            "n": int(len(value)),
            "probability_pearson": pearson,
            "probability_spearman": spearman,
            "prediction_disagreement_rate": float(np.mean(
                (p_left >= 0.5) != (p_right >= 0.5)
            )),
        }
        if fold is not None:
            row["fold"] = int(fold)
        rows.append(row)
    return rows


def assert_disjoint(train_index: np.ndarray, validation_index: np.ndarray) -> None:
    train = set(int(value) for value in np.asarray(train_index).reshape(-1))
    validation = set(int(value) for value in np.asarray(validation_index).reshape(-1))
    if train & validation:
        raise RuntimeError("outer validation rows leaked into committee training")


class FutureLabelGate:
    """Make the future-label access order executable rather than documentary."""

    def __init__(self, rows: list[dict]):
        self._rows = rows
        self._frozen = False

    def freeze_predictions_and_routes(self) -> None:
        self._frozen = True

    def labels(self, label_of: Callable[[dict], int]) -> np.ndarray:
        if not self._frozen:
            raise RuntimeError("future labels requested before routes/probabilities froze")
        return np.asarray([label_of(row) for row in self._rows], dtype=np.int64)
