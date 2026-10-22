#!/usr/bin/env python3
"""Artifact-independent uncertainty features and reliability router."""

from __future__ import annotations

import numpy as np

try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
except Exception as exc:  # pragma: no cover
    raise SystemExit("scikit-learn is required: " + str(exc))


PAPER9_NAMES = (
    "entropy",
    "kld_to_mean",
    "std",
    "max_second_max",
    "max_median",
    "max_mean",
    "median_mean",
    "mean_min",
    "median_min",
)

# Metric types implemented by MalCertain's public uncertainty extractor.
OFFICIAL7_NAMES = (
    "entropy",
    "kld_to_mean",
    "std",
    "max_second_max",
    "second_min_min",
    "max_min",
    "median_mean",
)


def _validated_probability_matrix(probability: np.ndarray) -> np.ndarray:
    matrix = np.asarray(probability, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 3:
        raise ValueError("probability matrix must be [samples, at least 3 views]")
    if np.any(~np.isfinite(matrix)):
        raise ValueError("probability matrix contains non-finite values")
    return np.clip(matrix, 1e-8, 1.0 - 1e-8)


def uncertainty_feature_dict(probability: np.ndarray) -> dict[str, np.ndarray]:
    """Compute paper-described and public-code uncertainty quantities."""
    p = _validated_probability_matrix(probability)
    ordered = np.sort(p, axis=1)
    mean = np.mean(p, axis=1)
    median = np.median(p, axis=1)
    minimum = ordered[:, 0]
    second_minimum = ordered[:, 1]
    second_maximum = ordered[:, -2]
    maximum = ordered[:, -1]

    # Average Bernoulli entropy of the view predictions.
    entropy = -np.mean(
        p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p), axis=1
    )
    # Average Bernoulli KL divergence from every view to their mean prediction.
    mean_2d = np.clip(mean[:, None], 1e-8, 1.0 - 1e-8)
    kld = np.mean(
        p * np.log2(p / mean_2d)
        + (1.0 - p) * np.log2((1.0 - p) / (1.0 - mean_2d)),
        axis=1,
    )
    return {
        "entropy": entropy,
        "kld_to_mean": kld,
        "std": np.std(p, axis=1, ddof=1),
        "max_second_max": maximum - second_maximum,
        "second_min_min": second_minimum - minimum,
        "max_min": maximum - minimum,
        "max_median": maximum - median,
        "max_mean": maximum - mean,
        "median_mean": median - mean,
        "mean_min": mean - minimum,
        "median_min": median - minimum,
    }


def router_features(
    probability: np.ndarray,
    variant: str,
) -> tuple[np.ndarray, list[str]]:
    """Return the main-view probability plus an uncertainty metric family."""
    p = _validated_probability_matrix(probability)
    values = uncertainty_feature_dict(p)
    if variant == "paper9":
        metric_names = PAPER9_NAMES
    elif variant == "official7":
        metric_names = OFFICIAL7_NAMES
    else:
        raise ValueError("unknown feature variant: " + variant)
    names = ["base_probability", *metric_names]
    matrix = np.column_stack([p[:, 0], *[values[name] for name in metric_names]])
    if np.any(~np.isfinite(matrix)):
        raise RuntimeError("non-finite router feature")
    return matrix, names


def fit_error_router(features: np.ndarray, error: np.ndarray, seed: int):
    """Fit a balanced SVM to rank the risk that Stage I is wrong."""
    error = np.asarray(error, dtype=np.int64)
    count = np.bincount(error, minlength=2)
    if int(np.min(count)) < 5:
        raise RuntimeError(
            "too few correct/error rows to train reliability router: "
            + repr(count.tolist())
        )
    model = Pipeline([
        ("scale", StandardScaler()),
        ("svm", SVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            probability=False,
            random_state=int(seed),
        )),
    ])
    model.fit(np.asarray(features, dtype=np.float64), error)
    return model


def error_risk_score(model, features: np.ndarray) -> np.ndarray:
    """Return a ranking score; larger means more likely to be misclassified."""
    score = np.asarray(model.decision_function(features), dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(score)):
        raise RuntimeError("non-finite SVM reliability score")
    return score


def top_fraction(score: np.ndarray, budget: float) -> np.ndarray:
    """Select an exact fixed budget using stable tie handling."""
    score = np.asarray(score, dtype=np.float64)
    if score.ndim != 1 or np.any(~np.isfinite(score)):
        raise ValueError("score must be a finite one-dimensional array")
    if not 0.0 < float(budget) < 1.0:
        raise ValueError("budget must be in (0,1)")
    count = max(1, min(len(score), int(round(float(budget) * len(score)))))
    order = np.argsort(score, kind="mergesort")
    result = np.zeros(len(score), dtype=bool)
    result[order[-count:]] = True
    return result
