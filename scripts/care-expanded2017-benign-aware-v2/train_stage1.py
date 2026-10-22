#!/usr/bin/env python3
"""Fit old/union/balanced formal P31-SVM arms; writes only EXPANDED_OUTPUT.

New 2017 benign APKs remain in the union. All fitting and calibration use
2017 labels only. New 2018–2022 labels are used solely in final metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from audit_new_apks import OUT, write_csv
from stage1_common import OLD_MODELS, SEED, fit_member, load_data, probability
from statistics import metrics


def save_pickle(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f: pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_or_fit(name, x, y, ids, *, balanced=False, jobs=2):
    path = OUT / 'models/stage1' / (name + '.pkl')
    if path.exists():
        with path.open('rb') as f: obj = pickle.load(f)
        if obj['train_sha'] != ids.tolist() or obj['class_weight'] != ('balanced' if balanced else None):
            raise RuntimeError('existing checkpoint contract changed: ' + str(path))
        return obj
    model = fit_member('SVM', x, y, ids, balanced=balanced, jobs=jobs)
    save_pickle(path, model)
    return model


def oof(name, x, y, ids, *, balanced=False, jobs=2):
    path = OUT / 'models/stage1' / (name + '_outer_oof.npz')
    if path.exists():
        with np.load(path, allow_pickle=False) as a:
            if not np.array_equal(a['ids'], ids): raise RuntimeError('OOF ids changed')
            return np.asarray(a['p'], float)
    p = np.full(len(y), np.nan)
    for fold, (fit, hold) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(x, y), 1):
        if set(ids[fit]) & set(ids[hold]): raise RuntimeError('outer-fold SHA leakage')
        model = fit_member('SVM', x[fit], y[fit], ids[fit], balanced=balanced,
                           seed=SEED+100000*fold, jobs=jobs)
        p[hold] = probability(model, x[hold])
        print(f'STAGE1_OOF {name} fold={fold}/5', flush=True)
    if not np.isfinite(p).all(): raise RuntimeError('OOF incomplete')
    with path.open('xb') as f: np.savez_compressed(f, ids=ids, p=p)
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-stage1', action='store_true', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 4: raise RuntimeError('workers outside safe range')
    data = load_data(require_all=False)
    old_ids, old_x, old_y = data['old_ids'], data['old_x'], data['old_y']
    ids, x, y = data['train_ids'], data['train_x'], data['train_y']
    test_ids, test_x, test_y = data['test_ids'], data['test_x'], data['test_y']
    if len(data['missing_train']) > 0 or len(data['missing_test']) > 0:
        (OUT / 'stage1_missing_p31.json').write_text(json.dumps({
            'train': data['missing_train'], 'test': data['missing_test']}, indent=2) + '\n')
    if len(ids) < 799 or len(x) != len(ids): raise RuntimeError('union P31 malformed')
    with (OLD_MODELS / 'SVM.pkl').open('rb') as f: old_checkpoint = pickle.load(f)
    s0 = old_checkpoint['model']
    if s0['train_sha'] != old_ids.tolist(): raise RuntimeError('old checkpoint/P31 order changed')
    s1 = load_or_fit('S1_OLD_REPLAY', old_x, old_y, old_ids, jobs=args.workers)
    replay_diff = float(np.max(np.abs(probability(s0, old_x)-probability(s1, old_x))))
    if replay_diff > 1e-8: raise RuntimeError(f'old Stage-I replay mismatch: {replay_diff}')
    print('STAGE1_OLD_REPLAY_PASS', replay_diff, flush=True)
    s2 = load_or_fit('S2_UNION', x, y, ids, jobs=args.workers)
    s3 = load_or_fit('S3_UNION_BALANCED', x, y, ids, balanced=True, jobs=args.workers)
    new_benign = np.flatnonzero((np.arange(len(ids)) >= 799) & (y == 0))
    rng = np.random.RandomState(SEED)
    chosen = np.sort(rng.choice(new_benign, min(400, len(new_benign)), replace=False))
    subset = np.concatenate([np.arange(799), chosen])
    s4 = load_or_fit('S4_FIXED_400_NEW_BENIGN', x[subset], y[subset], ids[subset], jobs=args.workers)
    arms = {'S0_OLD_FROZEN': s0, 'S1_OLD_REPLAY': s1, 'S2_UNION': s2,
            'S3_UNION_BALANCED': s3, 'S4_FIXED_400_NEW_BENIGN': s4}
    records = []; results = []
    years = np.array([int(r['year']) for r in data['test_rows']], int)
    for name, model in arms.items():
        p = probability(model, test_x)
        for sid, year, lab, score in zip(test_ids, years, test_y, p):
            records.append({'sha256': sid, 'year': int(year), 'label': int(lab),
                            'arm': name, 'probability': float(score),
                            'prediction': int(score >= .5)})
        for year in sorted(set(years)):
            mask = years == year
            results.append({'arm': name, 'scope': str(year), **metrics(test_y[mask], p[mask])})
        if len(test_y): results.append({'arm': name, 'scope': 'POOLED_NEW', **metrics(test_y, p)})
    write_csv(OUT / 'stage1_predictions.csv', records)
    write_csv(OUT / 'stage1_results.csv', results)
    src_results = []
    for name, xx, yy, ii, balanced in [('S1_OLD_REPLAY',old_x,old_y,old_ids,False),
            ('S2_UNION',x,y,ids,False),('S3_UNION_BALANCED',x,y,ids,True)]:
        pred = oof(name, xx, yy, ii, balanced=balanced, jobs=args.workers)
        src_results.append({'arm': name, 'scope': '2017_5F_OOF', **metrics(yy,pred)})
    write_csv(OUT / 'stage1_source_oof_results.csv', src_results)
    (OUT / 'stage1_training_audit.json').write_text(json.dumps({
        'status': 'PASS', 'old_replay_max_abs_difference': replay_diff,
        'old_train_n': 799, 'union_train_n': len(y), 'union_benign': int(sum(y == 0)),
        'union_malware': int(sum(y == 1)), 'fixed_subset_new_benign': len(chosen),
        'test_p31_n': len(test_ids), 'missing_train_p31': len(data['missing_train']),
        'missing_test_p31': len(data['missing_test']),
        'balanced_calibration': 'OOF raw SVM class_weight=balanced; sigmoid calibrated on original 2017 union class prior',
        'test_labels_used_for_training': False}, indent=2) + '\n')
    print('EXPANDED_STAGE1_DONE', len(ids), len(test_ids), flush=True)


if __name__ == '__main__': main()
