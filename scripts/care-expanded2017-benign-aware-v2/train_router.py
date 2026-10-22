#!/usr/bin/env python3
"""Fit formal four-member MeanOpposition controls on 2017 only.

Writes models, routing scores, annual routing metrics and label-free routed
lineages only below EXPANDED_OUTPUT. Future labels are read for metrics only.
"""
from __future__ import annotations

import argparse
import json
import pickle

import numpy as np

from audit_new_apks import OUT, write_csv
from stage1_common import (MEMBERS, OLD_MODELS, fit_member, load_data,
                           mean_opposition, probability, top25_by_year)


def checkpoint(path):
    with path.open('rb') as f: obj = pickle.load(f)
    return obj['model'] if 'model' in obj and 'raw' not in obj else obj


def new_aux(name, x, y, ids, jobs):
    path = OUT / 'models/router' / (name + '.pkl')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        obj = checkpoint(path)
        if obj['train_sha'] != ids.tolist(): raise RuntimeError('router checkpoint SHA mismatch')
        return obj
    obj = fit_member(name, x, y, ids, jobs=jobs)
    with path.open('xb') as f: pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return obj


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-router', action='store_true', required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()
    data = load_data(require_all=False)
    ids, x, y = data['train_ids'], data['train_x'], data['train_y']
    test_ids, tx, ty = data['test_ids'], data['test_x'], data['test_y']
    years = np.asarray([int(r['year']) for r in data['test_rows']], int)
    old = {name: checkpoint(OLD_MODELS / (name+'.pkl')) for name in MEMBERS}
    new = {name: new_aux(name, x, y, ids, args.workers) if name != 'SVM'
           else checkpoint(OUT / 'models/stage1/S2_UNION.pkl') for name in MEMBERS}
    balanced = checkpoint(OUT / 'models/stage1/S3_UNION_BALANCED.pkl')
    old_member = np.column_stack([probability(old[name], tx) for name in MEMBERS])
    new_member = np.column_stack([probability(new[name], tx) for name in MEMBERS])
    definitions = {
        'R0_OLD_STAGE1_OLD_AUX': (old_member, old_member[:, 3]),
        'R1_NEW_STAGE1_OLD_AUX': (old_member, new_member[:, 3]),
        'R2_NEW_STAGE1_NEW_AUX': (new_member, new_member[:, 3]),
        'R3_BALANCED_STAGE1_NEW_AUX': (new_member, probability(balanced, tx)),
    }
    records, result = [], []
    routed_any = np.zeros(len(test_ids), bool)
    for arm, (member, anchor) in definitions.items():
        score = mean_opposition(member, anchor)
        routed = top25_by_year(score, test_ids, years)
        routed_any |= routed
        for i, sid in enumerate(test_ids):
            records.append({'sha256': sid, 'year': int(years[i]), 'label': int(ty[i]),
                'arm': arm, 'stage1_probability': float(anchor[i]),
                'route_score': float(score[i]), 'routed': int(routed[i]),
                'stage1_error': int((anchor[i] >= .5) != ty[i])})
        for year in sorted(set(years)):
            ix = years == year
            bad = (anchor[ix] >= .5) != ty[ix]
            chosen = routed[ix]
            captured = int(np.sum(bad & chosen)); total = int(np.sum(bad)); n = int(np.sum(ix))
            all_fn = (anchor[ix] < .5) & (ty[ix] == 1)
            all_fp = (anchor[ix] >= .5) & (ty[ix] == 0)
            result.append({'arm': arm, 'year': int(year), 'n': n, 'routed': int(chosen.sum()),
                'stage1_errors': total, 'routed_errors': captured,
                'error_recall': captured/total if total else None,
                'route_precision': captured/int(chosen.sum()) if chosen.sum() else None,
                'routing_lift': (captured/int(chosen.sum()))/(total/n) if chosen.sum() and total else None,
                'unrouted_FN': int(np.sum(all_fn & ~chosen)),
                'unrouted_FP': int(np.sum(all_fp & ~chosen)),
                'routed_FN': int(np.sum(all_fn & chosen)),
                'routed_FP': int(np.sum(all_fp & chosen))})
    write_csv(OUT / 'router_predictions.csv', records)
    write_csv(OUT / 'router_results.csv', result)
    label_free = []
    for i, r in enumerate(data['test_rows']):
        if not routed_any[i]: continue
        label_free.append({'sha256': r['sha256'], 'absolute_path': r['apk_path'],
            'source_id': r['source_dataset'], 'lineage_id': r['source_dataset'],
            'metadata_year': r['year'], 'metadata_label': ''})
    write_csv(OUT / 'routed_new_future_label_free.csv', label_free)
    (OUT / 'router_training_audit.json').write_text(json.dumps({
        'status': 'PASS', 'members': list(MEMBERS),
        'route': 'SVM-anchored MeanOpposition Top25% separately in each year',
        'deep_extraction': 'union of all four pre-registered arm routes; no test-label routing',
        'train_2017_only': True, 'new_aux_train_n': len(y),
        'R1_definition': 'old LR/RF/XGB/SVM member predictions; union SVM anchor',
        'R3_definition': 'union auxiliary members; class-balanced union SVM anchor',
        'future_labels_used_for_training': False}, indent=2) + '\n')
    print('EXPANDED_ROUTER_DONE', len(test_ids), flush=True)


if __name__ == '__main__': main()
