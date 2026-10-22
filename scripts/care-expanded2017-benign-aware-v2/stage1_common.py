#!/usr/bin/env python3
"""Frozen P31-SVM/committee helpers for the isolated expanded-2017 experiment.

Writes nothing directly. The caller may write only below EXPANDED_OUTPUT.
The SVM construction, seed schedule and OOF sigmoid calibration reproduce the
existing CARE-Droid stage1_core and multi-year training implementation.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from audit_new_apks import OUT, PROJECT, read_csv

SEED = 20260904
MEMBERS = ('LR', 'RF', 'XGB', 'SVM')
SOURCE_P31 = PROJECT / 'artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz'
OLD_MODELS = PROJECT / 'outputs/care_multiyear_development_v1/models/stage1/S0R_2017'


def core_module():
    path = PROJECT / 'scripts/care-svm-train-window-expansion-v1/stage1_core.py'
    spec = importlib.util.spec_from_file_location('_expanded_formal_stage1_core', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize(value):
    return (value.decode() if isinstance(value, bytes) else str(value)).strip().lower()


def load_data(require_all=True):
    union = read_csv(OUT / 'union_2017_manifest.csv')
    tests = [r for r in read_csv(OUT / 'dataset_manifest.csv')
             if int(r['eligible_for_evaluation']) and int(r['year']) >= 2018]
    labels = {r['sha256'].lower(): int(r['normalized_label']) for r in union}
    with np.load(SOURCE_P31, allow_pickle=False) as a:
        old_ids = [normalize(x) for x in a['sha256']]
        old_x = np.asarray(a['role_prevalence'], float)
    if len(old_ids) != 799 or old_x.shape != (799, 31):
        raise RuntimeError('formal old 2017 P31 invalid')
    if any(s not in labels for s in old_ids):
        raise RuntimeError('formal old P31 SHA/label alignment missing')
    old_y = np.array([labels[s] for s in old_ids], int)
    shards = OUT / 'p31_shards'
    def new_x(rows):
        good_ids, good_x, good_y, good_rows, absent = [], [], [], [], []
        for r in rows:
            sid = r['sha256'].lower(); path = shards / (sid + '.npz')
            if not path.is_file():
                absent.append(sid); continue
            with np.load(path, allow_pickle=False) as a:
                x = np.asarray(a['p31'], float)
            if x.shape != (31,) or not np.isfinite(x).all():
                absent.append(sid); continue
            good_ids.append(sid); good_x.append(x); good_y.append(int(r['normalized_label'])); good_rows.append(r)
        return (np.asarray(good_ids), np.asarray(good_x, float).reshape(-1, 31),
                np.asarray(good_y, int), good_rows, absent)
    new_train = [r for r in union if r['source_dataset'] != 'OLD_2017']
    train_ids, train_x, train_y, train_rows, missing_train = new_x(new_train)
    test_ids, test_x, test_y, test_rows, missing_test = new_x(tests)
    if require_all and (missing_train or missing_test):
        raise RuntimeError(f'P31 incomplete train={len(missing_train)} test={len(missing_test)}; see extraction status')
    ids = np.concatenate([np.asarray(old_ids), train_ids])
    x = np.concatenate([old_x, train_x], axis=0)
    y = np.concatenate([old_y, train_y])
    if len(set(ids)) != len(ids): raise RuntimeError('train SHA collision')
    if set(ids) & set(test_ids): raise RuntimeError('train/test SHA leakage')
    return {'old_ids': np.asarray(old_ids), 'old_x': old_x, 'old_y': old_y,
            'train_ids': ids, 'train_x': x, 'train_y': y, 'new_train_ids': train_ids,
            'test_ids': test_ids, 'test_x': test_x, 'test_y': test_y, 'test_rows': test_rows,
            'missing_train': missing_train, 'missing_test': missing_test}


def fit_member(name, x, y, ids, *, balanced=False, seed=SEED, jobs=2):
    core = core_module()
    x = np.asarray(x, float); y = np.asarray(y, int); ids = np.asarray(ids)
    if set(np.unique(y)) != {0, 1}:
        raise RuntimeError('two-class fit required')
    if balanced and name != 'SVM':
        raise RuntimeError('balanced control defined only for SVM')
    mi = MEMBERS.index(name)
    oof = np.full(len(y), np.nan)
    splits = StratifiedKFold(3, shuffle=True, random_state=seed)
    for fold, (fit, hold) in enumerate(splits.split(x, y), 1):
        if set(ids[fit]) & set(ids[hold]): raise RuntimeError('calibration SHA leakage')
        model = core.make_member(name, seed + 10000 * mi + 100 * fold, jobs)
        if balanced: model.set_params(model__class_weight='balanced')
        model.fit(x[fit], y[fit]); oof[hold] = core.raw_score(model, x[hold])
    if not np.isfinite(oof).all(): raise RuntimeError('incomplete calibration OOF')
    calibrator = LogisticRegression(C=1e6, solver='lbfgs', max_iter=2000,
                                    random_state=seed + 10000 * mi + 999)
    calibrator.fit(oof.reshape(-1, 1), y)
    model = core.make_member(name, seed + 10000 * mi, jobs)
    if balanced: model.set_params(model__class_weight='balanced')
    model.fit(x, y)
    return {'raw': model, 'calibrator': calibrator, 'train_sha': ids.tolist(),
            'seed': seed, 'member': name, 'class_weight': 'balanced' if balanced else None,
            'calibration_prior': float(np.mean(y))}


def probability(model, x):
    core = core_module()
    raw = core.raw_score(model['raw'], np.asarray(x, float)).reshape(-1, 1)
    return np.clip(model['calibrator'].predict_proba(raw)[:, 1], 1e-8, 1 - 1e-8)


def mean_opposition(member, anchor):
    member = np.asarray(member, float); anchor = np.asarray(anchor, float)
    decision = anchor >= .5
    return np.where(decision[:, None], 1-member, member).mean(axis=1)


def top25_by_year(score, ids, years):
    score = np.asarray(score, float); ids = np.asarray(ids).astype(str); years = np.asarray(years, int)
    result = np.zeros(len(score), bool)
    for year in np.unique(years):
        pos = np.flatnonzero(years == year)
        order = np.lexsort((ids[pos], -score[pos]))
        result[pos[order[:int(round(len(pos)*.25))]]] = True
    return result
