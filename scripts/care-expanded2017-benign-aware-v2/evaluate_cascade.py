#!/usr/bin/env python3
"""Evaluate frozen U0–U4 cascades; writes only EXPANDED_OUTPUT.

All training and routes must already be frozen. No threshold, budget, model,
feature, or label definition is selected using these future-year results.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

from audit_new_apks import OUT, PROJECT, read_csv, write_csv
from statistics import metrics


ARMS = {
    'U0_OLD': ('S0_OLD_FROZEN', 'R0_OLD_STAGE1_OLD_AUX', 'C0_OLD_FROZEN'),
    'U1_NEW_STAGE1_OLD_ROUTER': ('S2_UNION', 'R1_NEW_STAGE1_OLD_AUX', 'C0_OLD_FROZEN'),
    'U2_NEW_STAGE1_NEW_ROUTER': ('S2_UNION', 'R2_NEW_STAGE1_NEW_AUX', 'C0_OLD_FROZEN'),
    'U3_NEW_STAGE2': ('S2_UNION', 'R2_NEW_STAGE1_NEW_AUX', 'C2_UNION'),
    'U4_BALANCED': ('S3_UNION_BALANCED', 'R3_BALANCED_STAGE1_NEW_AUX', 'C3_UNION_BALANCED'),
}


def conditional_auc(rows, kind):
    eligible = [r for r in rows if r['routed'] == 1 and r['stage1_case'] in kind]
    if len(eligible) < 2 or len({r['label'] for r in eligible}) < 2: return None
    return float(roc_auc_score([r['label'] for r in eligible],
                               [r['final_probability'] for r in eligible]))


def summarize(rows, arm, scope, cohort):
    y = np.asarray([r['label'] for r in rows], int)
    p = np.asarray([r['final_probability'] for r in rows], float)
    fixed = [r for r in rows if r['stage1_error'] and not r['final_error']]
    broken = [r for r in rows if not r['stage1_error'] and r['final_error']]
    raw = metrics(y, p)
    return {'arm': arm, 'scope': scope, 'cohort': cohort, **raw,
            'routed': sum(r['routed'] for r in rows),
            'routed_with_expert': sum(r['routed'] and r['expert_available'] for r in rows),
            'fixed': len(fixed), 'broken': len(broken), 'net': len(fixed)-len(broken),
            'fixed_FN': sum(r['stage1_case']=='FN' for r in fixed),
            'fixed_FP': sum(r['stage1_case']=='FP' for r in fixed),
            'broken_TN': sum(r['stage1_case']=='TN' for r in broken),
            'broken_TP': sum(r['stage1_case']=='TP' for r in broken),
            'FN_vs_TN_AUC': conditional_auc(rows, {'FN','TN'}),
            'TP_vs_FP_AUC': conditional_auc(rows, {'TP','FP'})}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate', action='store_true', required=True)
    args = parser.parse_args()
    for needed in ('stage1_training_audit.json','router_training_audit.json','stage2_training_audit.json'):
        if json.loads((OUT/needed).read_text())['status'] != 'PASS':
            raise RuntimeError('unfrozen prior phase: '+needed)
    stage1 = {(r['arm'],r['sha256']):r for r in read_csv(OUT/'stage1_predictions.csv')}
    routes = {(r['arm'],r['sha256']):r for r in read_csv(OUT/'router_predictions.csv')}
    expert = {r['sha256']:r for r in read_csv(OUT/'stage2_expert_predictions_label_free.csv')}
    manifest = {r['sha256']:r for r in read_csv(OUT/'dataset_manifest.csv')}
    prior = {r['sha256']:r for r in read_csv(PROJECT/'outputs/care_expanded2017_crossyear_v1/dataset_manifest.csv')}
    sha = sorted({sid for _,sid in stage1})
    records = []
    for arm, (sname,rname,cname) in ARMS.items():
        for sid in sha:
            s = stage1[(sname,sid)]; route = routes[(rname,sid)]
            year = int(s['year']); label = int(s['label']); base = float(s['probability'])
            routed = int(route['routed']); available = int(sid in expert)
            score = float(expert[sid][cname]) if routed and available else base
            pred0 = base >= .5; pred = score >= .5
            case = ('TP' if label else 'FP') if pred0 else ('FN' if label else 'TN')
            records.append({'arm': arm, 'sha256': sid, 'year': year, 'label': label,
                'source_dataset': manifest[sid]['source_dataset'],
                'novel_sha_vs_named_historical': int(prior[sid].get('novel_test','0') == '1'),
                'stage1_probability': base, 'router_score': float(route['route_score']),
                'routed': routed, 'expert_available': available,
                'stage2_model': cname, 'expert_probability': float(expert[sid][cname]) if available else None,
                'final_probability': score, 'final_prediction': int(pred),
                'stage1_case': case, 'stage1_error': int(pred0 != label),
                'final_error': int(pred != label),
                'fallback_reason': 'INCOMPLETE_STAGE2_ASSET' if routed and not available else ''})
    write_csv(OUT/'per_apk_predictions.csv', records)
    results = []
    for arm in ARMS:
        rr = [r for r in records if r['arm']==arm]
        for cohort, part in [('ALL',rr),('NOVEL_SHA',[r for r in rr if r['novel_sha_vs_named_historical']])]:
            for year in sorted({r['year'] for r in part}):
                results.append(summarize([r for r in part if r['year']==year],arm,str(year),cohort))
            results.append(summarize([r for r in part if 2018 <= r['year'] <= 2020],
                                     arm,'POOLED_2018_2020',cohort))
    write_csv(OUT/'cascade_results.csv', results)
    stage2 = [r for r in results if r['cohort']=='ALL' and r['scope'] in ('2018','2019','2020','POOLED_2018_2020')]
    write_csv(OUT/'stage2_results.csv', stage2)
    attribution = []
    for scope in ('2018','2019','2020','POOLED_2018_2020'):
        u2 = next(r for r in results if r['arm']=='U2_NEW_STAGE1_NEW_ROUTER' and r['scope']==scope and r['cohort']=='ALL')
        u3 = next(r for r in results if r['arm']=='U3_NEW_STAGE2' and r['scope']==scope and r['cohort']=='ALL')
        attribution.append({'scope': scope, 'n': u3['n'],
            'delta_BA_U3_minus_U2': None if u3['BA'] is None else u3['BA']-u2['BA'],
            'delta_FP': u3['FP']-u2['FP'], 'delta_FN': u3['FN']-u2['FN'],
            'delta_net': u3['net']-u2['net'],
            'interpretation': 'fixed Stage-I and Router; only Direct-C43 training differs'})
    write_csv(OUT/'module_attribution.csv', attribution)
    (OUT/'evaluation_audit.json').write_text(json.dumps({
        'status': 'PASS', 'threshold': .5, 'route_budget': .25,
        'test_labels_used_for_training': False, 'evaluation_years': sorted({int(r['year']) for r in records}),
        'future_2018_2021_historically_developed': True,
        'proxy_labels_not_behavior_truth': True,
        'single_class_metric_rule': 'BA/AUC undefined if one class absent'}, indent=2)+'\n')
    print('EXPANDED_CASCADE_EVALUATION_DONE', len(records), flush=True)


if __name__ == '__main__': main()
