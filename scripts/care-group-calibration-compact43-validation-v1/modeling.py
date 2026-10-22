#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def rf(seed):
    return RandomForestClassifier(
        n_estimators=300, max_features="sqrt", min_samples_leaf=2,
        class_weight="balanced", n_jobs=4, random_state=int(seed),
    )


def logit(probability):
    probability = np.clip(np.asarray(probability, float), 1e-7, 1 - 1e-7)
    return np.log(probability / (1 - probability))


def calibration_splits(y, groups, seed, n_splits=3):
    y = np.asarray(y, int)
    if groups is None:
        return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed)).split(np.zeros(len(y)), y))
    groups = np.asarray(groups, int)
    return list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(seed)).split(np.zeros(len(y)), y, groups))


def fit_calibrated_rf(x_train, y_train, x_target, seed, groups_train=None, sample_weight=None):
    x_train = np.asarray(x_train, float)
    y_train = np.asarray(y_train, int)
    x_target = np.asarray(x_target, float)
    raw_oof = np.full(len(y_train), np.nan)
    splits = calibration_splits(y_train, groups_train, seed, 3)
    for index, (fit, hold) in enumerate(splits, 1):
        model = rf(seed + 100 * index)
        fit_weight = None if sample_weight is None else np.asarray(sample_weight)[fit]
        model.fit(x_train[fit], y_train[fit], sample_weight=fit_weight)
        raw_oof[hold] = logit(model.predict_proba(x_train[hold])[:, 1])
    if not np.isfinite(raw_oof).all():
        raise RuntimeError("incomplete RF calibration OOF")
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000, random_state=seed + 999)
    calibrator.fit(raw_oof[:, None], y_train, sample_weight=sample_weight)
    model = rf(seed)
    model.fit(x_train, y_train, sample_weight=sample_weight)
    raw_target = logit(model.predict_proba(x_target)[:, 1])
    return np.clip(calibrator.predict_proba(raw_target[:, None])[:, 1], 1e-7, 1 - 1e-7)


def crossfit_calibrated_context(x, y, train_index, inner_fold, seed, groups=None, sample_weight=None):
    """Strict predictions for every outer-train row; its own label is excluded from RF and calibrator."""
    train_index = np.asarray(train_index, int)
    inner_fold = np.asarray(inner_fold, int)
    out = np.full(len(y), np.nan)
    for ordinal, held in enumerate(sorted(set(inner_fold[train_index].tolist())), 1):
        target = train_index[inner_fold[train_index] == held]
        fit = train_index[inner_fold[train_index] != held]
        fit_groups = None if groups is None else np.asarray(groups)[fit]
        fit_weight = None if sample_weight is None else np.asarray(sample_weight)[fit]
        out[target] = fit_calibrated_rf(
            x[fit], y[fit], x[target], seed + 10000 * ordinal,
            groups_train=fit_groups, sample_weight=fit_weight,
        )
    if not np.isfinite(out[train_index]).all():
        raise RuntimeError("incomplete strict nested Stage-I context")
    return out

