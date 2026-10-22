#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

MEMBERS = ("LR", "RF", "XGB", "SVM")


def make_member(name: str, seed: int, jobs: int = 4):
    if name == "LR":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=1, solver="liblinear", max_iter=5000,
                                         random_state=seed)),
        ])
    if name == "RF":
        return RandomForestClassifier(
            n_estimators=300, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced", n_jobs=jobs, random_state=seed,
        )
    if name == "XGB":
        return XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=.05, subsample=.8,
            colsample_bytree=.8, reg_lambda=1, objective="binary:logistic",
            eval_metric="logloss", tree_method="hist", n_jobs=jobs,
            random_state=seed, verbosity=0,
        )
    if name == "SVM":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", SVC(C=1, kernel="rbf", gamma="scale", probability=False,
                          random_state=seed)),
        ])
    raise KeyError(name)


def raw_score(model, values):
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(values), float).reshape(-1)
    probability = np.clip(
        model.predict_proba(values)[:, list(model.classes_).index(1)], 1e-8, 1 - 1e-8
    )
    return np.log(probability / (1 - probability))


def fit_calibrated_committee(x_train, y_train, x_target, seed: int,
                             inner_folds: int = 4, jobs: int = 4):
    """Fit four members and train-only Platt calibrators; target labels are absent."""
    x_train = np.asarray(x_train, float)
    y_train = np.asarray(y_train, int)
    x_target = np.asarray(x_target, float)
    splits = list(StratifiedKFold(
        inner_folds, shuffle=True, random_state=seed
    ).split(x_train, y_train))
    output = np.zeros((len(x_target), len(MEMBERS)), float)
    for member_index, name in enumerate(MEMBERS):
        calibration_oof = np.full(len(x_train), np.nan)
        for split_index, (fit_index, calibration_index) in enumerate(splits, 1):
            model = make_member(
                name, seed + 10000 * member_index + 100 * split_index, jobs
            )
            model.fit(x_train[fit_index], y_train[fit_index])
            calibration_oof[calibration_index] = raw_score(
                model, x_train[calibration_index]
            )
        if not np.isfinite(calibration_oof).all():
            raise RuntimeError("BLOCKED_BY_CONTRACT: incomplete calibration OOF")
        calibrator = LogisticRegression(
            C=1e6, solver="lbfgs", max_iter=2000,
            random_state=seed + 10000 * member_index + 999,
        )
        calibrator.fit(calibration_oof.reshape(-1, 1), y_train)
        model = make_member(name, seed + 10000 * member_index, jobs)
        model.fit(x_train, y_train)
        output[:, member_index] = np.clip(
            calibrator.predict_proba(
                raw_score(model, x_target).reshape(-1, 1)
            )[:, 1], 1e-8, 1 - 1e-8,
        )
    return output


def crossfit_mean_probability(x_train, y_train, seed: int, jobs: int = 4):
    """Nested OOF probabilities for Stage-II outer-train APKs only."""
    x_train = np.asarray(x_train, float)
    y_train = np.asarray(y_train, int)
    output = np.full(len(y_train), np.nan)
    splits = StratifiedKFold(4, shuffle=True, random_state=seed)
    for split_index, (fit_index, holdout_index) in enumerate(
            splits.split(x_train, y_train), 1):
        member_probability = fit_calibrated_committee(
            x_train[fit_index], y_train[fit_index], x_train[holdout_index],
            seed + 100000 * split_index, inner_folds=3, jobs=jobs,
        )
        output[holdout_index] = member_probability.mean(axis=1)
    if not np.isfinite(output).all():
        raise RuntimeError("BLOCKED_BY_CONTRACT: incomplete nested P31 outer-train OOF")
    return output


def mean_opposition(member_probability, mean_probability):
    member_probability = np.asarray(member_probability, float)
    decision = np.asarray(mean_probability, float) >= .5
    return np.where(decision[:, None], 1 - member_probability,
                    member_probability).mean(axis=1)


def foldwise_top25(score, fold, sha256):
    score = np.asarray(score, float)
    fold = np.asarray(fold, int)
    sha256 = np.asarray(sha256).astype(str)
    route = np.zeros(len(score), bool)
    for value in sorted(set(fold.tolist())):
        index = np.flatnonzero(fold == value)
        count = int(round(len(index) * .25))
        order = np.lexsort((sha256[index], -score[index]))
        route[index[order[:count]]] = True
    return route
