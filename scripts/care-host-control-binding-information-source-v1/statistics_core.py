#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
from common import auc

PRIMARY='B2_HOST_BINDING_PRIMARY'
COMPARATORS=('B1_HIDDEN_AGGREGATE','B3_WITHIN_APK_SHUFFLE__r00','B4_CAPACITY_PLACEBO__r00')
ENDPOINTS=('BA','FN_vs_TN_AUC','TP_vs_FP_AUC')

def endpoints(y,p,base,route):
    y=np.asarray(y);p=np.asarray(p);base=np.asarray(base);route=np.asarray(route,bool)
    q=p>=.5;first=base>=.5;before=first==y;after=q==y
    return dict(BA=.5*(float(np.mean(~q[y==0]))+float(np.mean(q[y==1]))),
        FN_vs_TN_AUC=auc(y[route&~first],p[route&~first]),TP_vs_FP_AUC=auc(y[route&first],p[route&first]),
        fixed=int((~before&after).sum()),broken=int((before&~after).sum()),net=int(after.sum()-before.sum()))

def strata_indices(y,folds,base,route):
    keys=np.column_stack([folds,y,np.asarray(route,int),(base>=.5).astype(int)])
    return [np.flatnonzero(np.all(keys==key,axis=1)) for key in sorted(set(map(tuple,keys.tolist())))]

def paired_intervals(y,folds,base,route,scores,repeats,seed):
    primary=[(a,e) for a in COMPARATORS for e in ENDPOINTS]+[('B0_DIRECT_C43','BA')]
    secondary=[(a,e) for a in ('B0_DIRECT_C43','B1_HIDDEN_AGGREGATE') for e in ('fixed','broken','net')]
    comparisons=primary+secondary;names={PRIMARY}|{a for a,e in comparisons}
    points={a:endpoints(y,scores[a],base,route) for a in names};samples={c:[] for c in comparisons}
    groups=strata_indices(y,folds,base,route);rng=np.random.default_rng(seed)
    for rep in range(repeats):
        ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        b={a:endpoints(y[ix],scores[a][ix],base[ix],route[ix]) for a in names}
        for a,e in comparisons:
            if b[PRIMARY][e] is not None and b[a][e] is not None:samples[(a,e)].append(b[PRIMARY][e]-b[a][e])
        if repeats>=1000 and (rep+1)%250==0:print(f'HBI_PAIRED_BOOTSTRAP {rep+1}/{repeats}',flush=True)
    rows=[];alpha=.05/len(primary)
    for a,e in comparisons:
        v=np.asarray(samples[(a,e)],float);is_primary=(a,e) in primary
        estimate=points[PRIMARY][e]-points[a][e] if points[PRIMARY][e] is not None and points[a][e] is not None else None
        lo,hi=np.quantile(v,[.025,.975]) if len(v) else (None,None)
        al,ah=np.quantile(v,[alpha/2,1-alpha/2]) if len(v) and is_primary else (None,None)
        rows.append(dict(comparator=a,endpoint=e,difference=estimate,paired_95_low=float(lo) if lo is not None else None,
            paired_95_high=float(hi) if hi is not None else None,multiplicity_adjusted_low=float(al) if al is not None else None,
            multiplicity_adjusted_high=float(ah) if ah is not None else None,primary_family=is_primary,
            family_size=len(primary) if is_primary else None,valid_bootstraps=len(v),unit='APK',
            inference_scope='fixed fitted models and historical cohort; no retraining/family dependence uncertainty'))
    return rows

def gate(points,fold_points,fold_metrics,intervals,null95,p,valid):
    im={(r['comparator'],r['endpoint']):r for r in intervals};direction={};fold_support={}
    for side in ('FN_vs_TN_AUC','TP_vs_FP_AUC'):
        available=[f for f in range(1,6) if fold_points[(PRIMARY,f)][side] is not None and fold_points[('B1_HIDDEN_AGGREGATE',f)][side] is not None]
        positive=sum(fold_points[(PRIMARY,f)][side]>fold_points[('B1_HIDDEN_AGGREGATE',f)][side] for f in available)
        fold_support[side]=dict(evaluable_folds=len(available),positive_folds_vs_aggregate=positive)
        g=p['information_gate'];real=points[PRIMARY][side];aggregate=points['B1_HIDDEN_AGGREGATE'][side]
        flags=dict(real_at_least_aggregate_plus_0_02=real is not None and aggregate is not None and real>=aggregate+g['minimum_auc_gain_vs_B1_and_each_null95'],
            real_at_least_each_null95_plus_0_02=real is not None and all(null95[a][side] is not None and real>=null95[a][side]+g['minimum_auc_gain_vs_B1_and_each_null95'] for a in null95),
            evaluable_folds=len(available)>=g['minimum_evaluable_folds_per_side'],
            majority_folds=positive>=g['minimum_positive_folds_vs_B1_per_side'],
            adjusted_AUC_differences_positive=all(im[(a,side)]['multiplicity_adjusted_low'] is not None and im[(a,side)]['multiplicity_adjusted_low']>0 for a in COMPARATORS))
        direction[side]=dict(supported=all(flags.values()),criteria=flags)
    nets=[fold_metrics[(PRIMARY,f)]['net']-fold_metrics[('B1_HIDDEN_AGGREGATE',f)]['net'] for f in range(1,6)]
    real=points[PRIMARY];g=p['integration_gate']
    checks=dict(net_above_direct_and_aggregate=all(real['net']>points[a]['net'] for a in ('B0_DIRECT_C43','B1_HIDDEN_AGGREGATE')),
        broken_not_above_direct_or_aggregate=all(real['broken']<=points[a]['broken'] for a in ('B0_DIRECT_C43','B1_HIDDEN_AGGREGATE')),
        at_least_4_folds_net_nondecreasing=sum(v>=0 for v in nets)>=g['minimum_folds_net_nondecreasing_vs_B1'],
        at_least_3_folds_net_improving=sum(v>0 for v in nets)>=g['minimum_folds_net_strictly_better_vs_B1'],
        BA_above_each_null95=all(real['BA']>null95[a]['BA'] for a in null95),
        adjusted_BA_differences_positive=all(im[(a,'BA')]['multiplicity_adjusted_low'] is not None and im[(a,'BA')]['multiplicity_adjusted_low']>0 for a in COMPARATORS+('B0_DIRECT_C43',)))
    both=all(v['supported'] for v in direction.values());one=any(v['supported'] for v in direction.values())
    if not valid:status='HOST_BINDING_INFORMATION_INCONCLUSIVE'
    elif both and all(checks.values()):status='HOST_BINDING_INFORMATION_SUPPORTED_SOURCE_ONLY'
    elif both:status='HOST_BINDING_DIRECTION_SUPPORTED_INTEGRATION_NOT_CONFIRMED'
    elif one:status='PARTIALLY_SUPPORTED'
    else:status='HOST_BINDING_INFORMATION_NOT_SUPPORTED'
    return dict(status=status,direction=direction,fold_direction_support=fold_support,integration_criteria=checks,
        permit_specifying_separate_frozen_diagnostic=bool(valid and both and all(checks.values())))
