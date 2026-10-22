#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

MEMBERS = ("LR", "RF", "XGB", "SVM")


def make_member(name, seed, jobs=4):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    if name == "LR":
        return Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=1, solver="liblinear", max_iter=5000, random_state=int(seed)))])
    if name == "RF":
        return RandomForestClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=2, class_weight="balanced", n_jobs=jobs, random_state=int(seed))
    if name == "XGB":
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=.05, subsample=.8, colsample_bytree=.8, reg_lambda=1, objective="binary:logistic", eval_metric="logloss", tree_method="hist", n_jobs=jobs, random_state=int(seed), verbosity=0)
    if name == "SVM":
        return Pipeline([("scale", StandardScaler()), ("model", SVC(C=1, kernel="rbf", gamma="scale", probability=False, random_state=int(seed)))])
    raise KeyError(name)


def raw_score(model, values):
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(values), float).reshape(-1)
    p = np.clip(model.predict_proba(values)[:, list(model.classes_).index(1)], 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p))


def fit_calibrated_committee(x_train, y_train, x_target, seed, inner_folds=3, jobs=4):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    x_train = np.asarray(x_train, float); y_train = np.asarray(y_train, int); x_target = np.asarray(x_target, float)
    splits = list(StratifiedKFold(inner_folds, shuffle=True, random_state=int(seed)).split(x_train, y_train))
    out = np.zeros((len(x_target), 4), float)
    for member_index, name in enumerate(MEMBERS):
        oof = np.full(len(y_train), np.nan)
        for split_index, (fit, calibration) in enumerate(splits, 1):
            model = make_member(name, int(seed) + 10000 * member_index + 100 * split_index, jobs)
            model.fit(x_train[fit], y_train[fit])
            oof[calibration] = raw_score(model, x_train[calibration])
        if not np.isfinite(oof).all():
            raise RuntimeError("BLOCKED_BY_CONTRACT: incomplete calibration OOF")
        calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000, random_state=int(seed) + 10000 * member_index + 999)
        calibrator.fit(oof.reshape(-1, 1), y_train)
        model = make_member(name, int(seed) + 10000 * member_index, jobs)
        model.fit(x_train, y_train)
        out[:, member_index] = np.clip(calibrator.predict_proba(raw_score(model, x_target).reshape(-1, 1))[:, 1], 1e-8, 1 - 1e-8)
    return out


def crossfit_rf_probability(x, y, seed, jobs=4):
    from sklearn.model_selection import StratifiedKFold
    x = np.asarray(x, float); y = np.asarray(y, int); out = np.full(len(y), np.nan)
    for split_index, (fit, hold) in enumerate(StratifiedKFold(4, shuffle=True, random_state=int(seed)).split(x, y), 1):
        members = fit_calibrated_committee(x[fit], y[fit], x[hold], int(seed) + 100000 * split_index, inner_folds=3, jobs=jobs)
        out[hold] = members[:, MEMBERS.index("RF")]
    if not np.isfinite(out).all():
        raise RuntimeError("BLOCKED_BY_CONTRACT: incomplete nested RF OOF")
    return out


def mean_opposition(member_probability, rf_probability):
    member_probability = np.asarray(member_probability, float)
    decision = np.asarray(rf_probability, float) >= .5
    return np.where(decision[:, None], 1 - member_probability, member_probability).mean(axis=1)


def top25(score, sha):
    score = np.asarray(score, float); sha = np.asarray(sha).astype(str)
    count = int(round(len(score) * .25))
    order = np.lexsort((sha, -score))
    route = np.zeros(len(score), bool); route[order[:count]] = True
    return route
