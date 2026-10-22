#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

from common import FUTURE_LABELS, OUT, norm, read_csv, save_json, write_csv

METHOD = 'P31Mean__DI25__C43RelationQuery__Bounded'


def metrics(y, p):
    y = np.asarray(y, int); p = np.asarray(p, float); pred = p >= .5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'BA': float(balanced_accuracy_score(y, pred)), 'AUC': float(roc_auc_score(y, p)),
            'F1': float(f1_score(y, pred, zero_division=0)), 'TN': int(tn), 'FP': int(fp),
            'FN': int(fn), 'TP': int(tp), 'errors': int(np.sum(pred != y))}


def find_prediction_file():
    exact = OUT / 'arm/P31_C43_RQ/future_predictions_diagnostic.csv'
    if exact.is_file(): return exact
    found = sorted((OUT / 'arm/P31_C43_RQ').glob('**/future_predictions_diagnostic.csv'))
    if len(found) == 1: return found[0]
    raise RuntimeError(f'BLOCKED_BY_CONTRACT: future prediction file ambiguity: {found}')


def main():
    ref = read_csv(OUT / 'preflight/p31_reference/future_predictions_diagnostic.csv')
    pred = read_csv(find_prediction_file())
    labels = read_csv(FUTURE_LABELS)
    label_map = {norm(r['sha256']): (int(float(r.get('year', 0))), int(float(r.get('y', r.get('label')))))
                 for r in labels if norm(r.get('sha256')) and str(r.get('y', r.get('label', ''))).strip() != ''}
    rmap = {norm(r['sha256']): r for r in ref}
    pmap = {norm(r['sha256']): r for r in pred}
    ids = sorted(rmap)
    if len(ids) != 1989 or set(ids) - set(pmap) or set(ids) - set(label_map):
        raise RuntimeError('BLOCKED_BY_CONTRACT: future reference/prediction/label SHA mismatch')

    columns = set(pred[0])
    final_col = METHOD + '__prob'
    if final_col not in columns:
        candidates = [c for c in columns if c.endswith('__Bounded__prob') and 'expert' not in c.casefold()]
        if len(candidates) != 1:
            raise RuntimeError(f'BLOCKED_BY_CONTRACT: cannot identify final bounded probability column: {candidates}')
        final_col = candidates[0]
    expert_col = METHOD + '__expert_prob' if METHOD + '__expert_prob' in columns else None

    records = []
    for sid in ids:
        rr, pp = rmap[sid], pmap[sid]
        year, y = label_map[sid]
        base = float(rr['P31_MEAN_COMMITTEE__prob'])
        route = int(float(rr['P31_DI25__routed'])) == 1
        final = float(pp[final_col])
        if 'Global24__prob' in pp and abs(float(pp['Global24__prob']) - base) > 1e-10:
            raise RuntimeError(f'BLOCKED_BY_CONTRACT: future runner changed P31 base probability for {sid}')
        if 'A25__routed' in pp and (int(float(pp['A25__routed'])) == 1) != route:
            raise RuntimeError(f'BLOCKED_BY_CONTRACT: future runner changed P31-DI25 route for {sid}')
        if not route and abs(final - base) > 1e-10:
            raise RuntimeError(f'BLOCKED_BY_CONTRACT: correction applied outside P31-DI25 for {sid}')
        bpred = base >= .5; fpred = final >= .5
        status = ('fixed' if bpred != y and fpred == y else 'broken' if bpred == y and fpred != y else
                  'same_correct' if bpred == y else 'same_wrong')
        row = {'sha256': sid, 'year': year, 'label': y, 'P31_prob': base,
               'P31_DI25_score': float(rr['P31_DI25__score']), 'P31_DI25_routed': int(route),
               'final_prob': final, 'status': status}
        if expert_col:
            row['C43_expert_prob'] = float(pp[expert_col])
        records.append(row)

    annual = []
    for year in (2018, 2019, 2020, 2021):
        subset = [r for r in records if r['year'] == year]
        y = np.asarray([r['label'] for r in subset]); base = np.asarray([r['P31_prob'] for r in subset]); final = np.asarray([r['final_prob'] for r in subset])
        route = np.asarray([r['P31_DI25_routed'] == 1 for r in subset]); err = (base >= .5) != y
        fixed = sum(r['status'] == 'fixed' for r in subset); broken = sum(r['status'] == 'broken' for r in subset)
        bm, fm = metrics(y, base), metrics(y, final)
        annual.append({'year': year, 'n': len(subset), 'routed': int(route.sum()),
                       'stage1_errors': int(err.sum()), 'routed_stage1_errors': int(np.sum(route & err)),
                       'route_error_recall': float(np.sum(route & err) / max(int(err.sum()), 1)),
                       'route_precision': float(np.sum(route & err) / max(int(route.sum()), 1)),
                       'stage1_BA': bm['BA'], 'stage1_AUC': bm['AUC'], 'stage1_F1': bm['F1'],
                       'final_BA': fm['BA'], 'final_AUC': fm['AUC'], 'final_F1': fm['F1'],
                       'fixed': fixed, 'broken': broken, 'net': fixed - broken,
                       'delta_BA': fm['BA'] - bm['BA']})

    y = np.asarray([r['label'] for r in records]); base = np.asarray([r['P31_prob'] for r in records]); final = np.asarray([r['final_prob'] for r in records])
    route = np.asarray([r['P31_DI25_routed'] == 1 for r in records]); err = (base >= .5) != y
    bm, fm = metrics(y, base), metrics(y, final)
    fixed = sum(r['status'] == 'fixed' for r in records); broken = sum(r['status'] == 'broken' for r in records)
    pooled = {'year': 'pooled', 'n': len(records), 'routed': int(route.sum()), 'stage1_errors': int(err.sum()),
              'routed_stage1_errors': int(np.sum(route & err)),
              'route_error_recall': float(np.sum(route & err) / max(int(err.sum()), 1)),
              'route_precision': float(np.sum(route & err) / max(int(route.sum()), 1)),
              'stage1_BA': bm['BA'], 'stage1_AUC': bm['AUC'], 'stage1_F1': bm['F1'],
              'final_BA': fm['BA'], 'final_AUC': fm['AUC'], 'final_F1': fm['F1'],
              'fixed': fixed, 'broken': broken, 'net': fixed - broken, 'delta_BA': fm['BA'] - bm['BA']}

    results = OUT / 'results'
    write_csv(results / 'future_annual_summary.csv', annual)
    write_csv(results / 'future_pooled_summary.csv', [pooled])
    write_csv(results / 'future_paired_changes.csv', records)
    save_json(results / 'future_summary_contract.json', {
        'diagnostic_only': True,
        'future_years': [2018, 2019, 2020, 2021],
        'stage1': 'P31 Mean Committee trained on all 2017 only',
        'router': 'P31 MeanOpposition year-wise Top25%',
        'stage2': 'C43_RELATION_QUERY bounded correction',
        'max_delta': 2.25,
        'future_results_must_not_be_used_for_model_selection_or_threshold_tuning': True,
        'final_probability_column': final_col,
    })
    lines = ['CARE-Droid P31 -> P31-DI25 -> C43-RQ future diagnostic', '',
             '2018-2021 are diagnostic only. P31 predictions/routes were frozen before future labels were unsealed.', '']
    for r in annual:
        lines.append(f"{r['year']}: n={r['n']} P31_BA={r['stage1_BA']:.6f} routed={r['routed']} "
                     f"error_recall={r['route_error_recall']:.4f} precision={r['route_precision']:.4f} "
                     f"Final_BA={r['final_BA']:.6f} fixed={r['fixed']} broken={r['broken']} net={r['net']} delta_BA={r['delta_BA']:+.6f}")
    lines += ['', f"POOLED: n={pooled['n']} P31_BA={pooled['stage1_BA']:.6f} P31_AUC={pooled['stage1_AUC']:.6f} "
              f"routed={pooled['routed']} error_recall={pooled['route_error_recall']:.4f} precision={pooled['route_precision']:.4f}",
              f"FINAL: BA={pooled['final_BA']:.6f} AUC={pooled['final_AUC']:.6f} F1={pooled['final_F1']:.6f} "
              f"fixed={pooled['fixed']} broken={pooled['broken']} net={pooled['net']} delta_BA={pooled['delta_BA']:+.6f}", '',
              'Interpretation boundary: do not use these future results to choose the router, threshold, representation, or correction architecture.']
    (results / 'REPORT.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('P31_C43_FUTURE_SUMMARY_PASS')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
