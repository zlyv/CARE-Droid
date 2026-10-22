#!/usr/bin/env python3
"""Leakage-safe Stage-I probability protocol for the future formal runner."""
from __future__ import annotations

import numpy as np

from source_contract import blocked


def inner_oof_stage1(outer_train_indices, labels, fit_predict_global24, folds: int = 4,
                     split_seed: int = 20260901, model_seed: int = 0,
                     splitter_factory=None):
    """Generate Stage-I probabilities without using frozen outer-OOF values.

    fit_predict_global24 receives `(fit_global_indices, validation_global_indices,
    model_seed)` and must fit all preprocessing/model state on fit indices only.
    """
    outer_train = np.asarray(outer_train_indices, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int64)[outer_train]
    if len(outer_train) == 0 or set(np.unique(y).tolist()) != {0, 1}:
        raise blocked("invalid outer-train cohort for inner-OOF Stage-I")
    output = np.full(len(outer_train), np.nan, dtype=np.float64)
    assignment = np.zeros(len(outer_train), dtype=np.int64)
    if splitter_factory is None:
        # Lazy import keeps source-contract selftests independent of the server ML stack.
        from sklearn.model_selection import StratifiedKFold
        splitter = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(split_seed))
    else:
        splitter = splitter_factory(int(folds), int(split_seed))
    for inner_fold, (fit_local, validation_local) in enumerate(splitter.split(np.zeros(len(y)), y), start=1):
        fit_global = outer_train[fit_local]
        validation_global = outer_train[validation_local]
        if set(fit_global.tolist()) & set(validation_global.tolist()):
            raise blocked("inner Stage-I fit/validation overlap")
        probability = np.asarray(
            fit_predict_global24(fit_global, validation_global, int(model_seed) + inner_fold),
            dtype=np.float64,
        )
        if probability.shape != (len(validation_local),) or not np.isfinite(probability).all():
            raise blocked("inner Stage-I predictor returned invalid probabilities")
        output[validation_local] = probability
        assignment[validation_local] += 1
    if not np.isfinite(output).all() or not np.all(assignment == 1):
        raise blocked("inner-OOF Stage-I coverage is not exactly once")
    return output


def outer_fold_indices(folds, fold: int):
    values = np.asarray(folds, dtype=np.int64)
    train = np.flatnonzero(values != int(fold))
    validation = np.flatnonzero(values == int(fold))
    if set(train.tolist()) & set(validation.tolist()) or len(train) + len(validation) != len(values):
        raise blocked("outer fold partition failed")
    return train, validation
