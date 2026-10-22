#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from common import (OUT, SOURCE_P31, SOURCE_REFERENCE, SOURCE_ROUTER, discover_future_p31_cache,
                    file_sha256, load_future_cohort, norm, read_csv, save_json, write_csv)
from p31_core import MEMBERS, fit_calibrated_committee, mean_opposition

SEED = 20260925


def yearwise_top25(score, years, ids):
    score = np.asarray(score, float)
    years = np.asarray(years, int)
    ids = np.asarray(ids).astype(str)
    route = np.zeros(len(score), bool)
    for year in (2018, 2019, 2020, 2021):
        idx = np.flatnonzero(years == year)
        count = int(round(len(idx) * .25))
        order = np.lexsort((ids[idx], -score[idx]))
        route[idx[order[:count]]] = True
    return route


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--future-p31-cache', default=None)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--rebuild', action='store_true')
    args = ap.parse_args()

    ref_dir = OUT / 'preflight/p31_reference'
    future_ref = ref_dir / 'future_predictions_diagnostic.csv'
    member_file = OUT / 'preflight/p31_future_member_predictions_label_free.csv'
    if future_ref.is_file() and member_file.is_file() and not args.rebuild:
        print(f'[resume] P31 future reference already exists: {future_ref}')
        return

    if not SOURCE_P31.is_file() or not SOURCE_REFERENCE.is_file() or not SOURCE_ROUTER.is_file():
        raise RuntimeError('BLOCKED_BY_CONTRACT: source P31 experiment/artifacts are incomplete; run the strict P31 source experiment first')
    future_cache = discover_future_p31_cache(args.future_p31_cache)
    cohort_path, cohort = load_future_cohort()
    future_ids = np.asarray([sid for sid, _ in cohort])
    years = np.asarray([year for _, year in cohort], int)

    source_router = read_csv(SOURCE_ROUTER)
    source_label = {norm(r['sha256']): int(float(r['label'])) for r in source_router}
    with np.load(SOURCE_P31, allow_pickle=False) as z:
        source_ids = np.asarray([norm(x) for x in z['sha256']])
        source_x = np.asarray(z['role_prevalence'], float)
    if source_x.shape != (799, 31) or len(set(source_ids.tolist())) != 799:
        raise RuntimeError(f'BLOCKED_BY_CONTRACT: source P31 shape changed: {source_x.shape}')
    if set(source_ids.tolist()) != set(source_label):
        raise RuntimeError('BLOCKED_BY_CONTRACT: source P31/source-label SHA mismatch')
    y = np.asarray([source_label[s] for s in source_ids], int)

    with np.load(future_cache, allow_pickle=False) as z:
        all_ids = np.asarray([norm(x) for x in z['sha256']])
        all_x = np.asarray(z['role_prevalence'], float)
    if all_x.ndim != 2 or all_x.shape[1] != 31 or len(all_ids) != len(all_x):
        raise RuntimeError(f'BLOCKED_BY_CONTRACT: future P31 cache has invalid shape: {all_x.shape}')
    pos = {sid: i for i, sid in enumerate(all_ids)}
    missing = [sid for sid in future_ids if sid not in pos]
    if missing:
        raise RuntimeError(f'BLOCKED_BY_CONTRACT: future P31 cache missing {len(missing)} cohort SHA; first={missing[:3]}')
    future_x = all_x[[pos[sid] for sid in future_ids]]

    # Full-source Stage-I fit. Future labels are not loaded or used here.
    member_p = fit_calibrated_committee(source_x, y, future_x, seed=SEED, inner_folds=4, jobs=args.jobs)
    p = member_p.mean(axis=1)
    score = mean_opposition(member_p, p)
    route = yearwise_top25(score, years, future_ids)

    expected_route = {2018: 198, 2019: 100, 2020: 99, 2021: 100}
    observed_route = {year: int(route[years == year].sum()) for year in expected_route}
    if observed_route != expected_route or int(route.sum()) != 497:
        raise RuntimeError(f'BLOCKED_BY_CONTRACT: P31 DI25 future budget mismatch: {observed_route}, total={int(route.sum())}')

    rows = []
    member_rows = []
    for i, sid in enumerate(future_ids):
        rows.append({
            'sha256': sid,
            'year': int(years[i]),
            # Compatibility fields consumed by the frozen future runner.
            'Global24__prob': float(p[i]),
            'A25__score': float(score[i]),
            'A25__routed': int(route[i]),
            'Global24__A25__JointS7_ordinary__prob': float(p[i]),
            # Explicit semantics.
            'P31_MEAN_COMMITTEE__prob': float(p[i]),
            'P31_DI25__score': float(score[i]),
            'P31_DI25__routed': int(route[i]),
            'diagnostic_only': True,
        })
        mr = {'sha256': sid, 'year': int(years[i]), 'P31_MEAN_COMMITTEE__prob': float(p[i]),
              'P31_DI25__score': float(score[i]), 'P31_DI25__routed': int(route[i])}
        for j, name in enumerate(MEMBERS):
            mr[f'p_P31__{name}'] = float(member_p[i, j])
        member_rows.append(mr)

    ref_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_REFERENCE, ref_dir / 'source_predictions.csv')
    write_csv(future_ref, rows)
    write_csv(member_file, member_rows)
    contract = {
        'experiment': 'P31 full-source Stage-I -> year-wise P31-DI25 future reference',
        'stage1_representation': 'P31 only; 31-D sensitive-role method prevalence',
        'stage1_definition': 'mean of calibrated LR/RF/XGB/SVM',
        'source_training_rows': 799,
        'source_training_labels': '2017 only',
        'future_rows': 1989,
        'future_labels_loaded_for_stage1_or_routing': False,
        'future_p31_cache': str(future_cache),
        'future_label_free_cohort': str(cohort_path),
        'full_source_seed': SEED,
        'router': 'MeanOpposition, Top25% independently within each future year',
        'route_by_year': observed_route,
        'route_total': int(route.sum()),
        'future_reference_sha256': file_sha256(future_ref),
    }
    save_json(OUT / 'preflight/p31_future_contract.json', contract)
    print('P31_FUTURE_REFERENCE_PASS')
    print(f'future_p31_cache={future_cache}')
    print(f'future_rows=1989 route_total=497 route_by_year={observed_route}')
    print('future_labels_loaded_for_stage1_or_routing=false')


if __name__ == '__main__':
    main()
