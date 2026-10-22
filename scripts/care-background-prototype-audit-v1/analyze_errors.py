#!/usr/bin/env python3
"""Join fixed error/pair cohorts to frozen per-APK background responses.

All labels are diagnostic validation labels. Pair definitions are pre-existing;
no label, error status or future-year outcome changes the Prototype Bank.
Writes only the new audit output directory.
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict

import numpy as np
from scipy.stats import mannwhitneyu

from audit_assets import OUT,ROOT,read_csv,write_csv,write_json

METRICS=('mean_external','mean_internal','mean_commonness','mean_residual',
         'high_support_shared_count','benign_associated_external_count',
         'external_shared_overlap_fraction','shared_supported_count',
         'external_supported_count')


def num(row,key):
    value=row.get(key,'')
    return float(value) if value not in ('',None,'None','nan') else np.nan


def bh(pvalues):
    valid=[(i,p) for i,p in enumerate(pvalues) if p is not None and np.isfinite(p)]
    out=[None]*len(pvalues);m=len(valid)
    if not m:return out
    ordered=sorted(valid,key=lambda x:x[1]);minimum=1.
    for rank in range(m,0,-1):
        i,p=ordered[rank-1]
        minimum=min(minimum,p*m/rank)
        out[i]=minimum
    return out


def pair_2020(rows):
    pairs=read_csv(ROOT/'outputs/care-2020-fn-tn-dex-review-v2/matched_pairs_restricted.csv')
    if len(pairs)!=22:raise RuntimeError('pre-existing 2020 matched pair count changed')
    out=[];lookup={(r['tag'],int(r['seed']),r['sha256']):r for r in rows if r['split']=='VALIDATION'}
    for pidx,p in enumerate(pairs,1):
        for seed in (900000,900001,900002):
            fn=lookup.get(('T3_C1',seed,p['fn_sha256']))
            tn=lookup.get(('T3_C1',seed,p['tn_sha256']))
            if fn is None or tn is None:raise RuntimeError('frozen pair missing in bank cache')
            if fn['category']!='UNFIXED_FN' or tn['label']!='0':
                raise RuntimeError('frozen pair/error contract changed')
            z={'pair_id':pidx,'tag':'T3_C1','seed':seed,'year':2020,
               'fn_sha256':fn['sha256'],'tn_sha256':tn['sha256'],
               'match_stage1_probability_difference':p['stage1_probability_difference'],
               'match_host_count_ratio':p['host_count_ratio'],
               'match_p31_standardized_cosine_distance':p['p31_standardized_cosine_distance'],
               'fn_hosts':fn['host_count'],'tn_hosts':tn['host_count']}
            for m in METRICS:
                z['FN_'+m]=fn[m];z['TN_'+m]=tn[m]
                z['FN_minus_TN_'+m]=(num(fn,m)-num(tn,m)) if np.isfinite(num(fn,m)) and np.isfinite(num(tn,m)) else None
            out.append(z)
    return out


def matched_interval(rows):
    # Average correlated seeds first; the independent unit is the matched APK pair.
    result=[];rng=np.random.RandomState(20261016)
    for metric in METRICS:
        bypair=defaultdict(list)
        for r in rows:
            value=num(r,'FN_minus_TN_'+metric)
            if np.isfinite(value):bypair[int(r['pair_id'])].append(value)
        values=np.asarray([np.mean(v) for _,v in sorted(bypair.items())],float)
        if len(values)<10:
            result.append({'comparison':'2020_FN_vs_matched_TN','metric':metric,'pairs_evaluable':len(values),
                           'status':'INSUFFICIENT_EVALUABLE_PAIRS'})
            continue
        observed=float(values.mean())
        boots=rng.choice(values,(2000,len(values)),replace=True).mean(axis=1)
        signed=(rng.choice((-1.,1.),(5000,len(values)))*values).mean(axis=1)
        p=(1+int(np.sum(np.abs(signed)>=abs(observed))))/(len(signed)+1)
        result.append({'comparison':'2020_FN_vs_matched_TN','metric':metric,
            'pairs_evaluable':len(values),'seed_repeats_not_independent':3,
            'mean_FN_minus_TN':observed,'CI95_low':float(np.quantile(boots,.025)),
            'CI95_high':float(np.quantile(boots,.975)),'two_sided_sign_flip_p':p,
            'status':'DESCRIPTIVE_DEVELOPMENT_SET'})
    adj=bh([r.get('two_sided_sign_flip_p') for r in result])
    for r,q in zip(result,adj):r['BH_q_across_predeclared_metrics']=q
    return result


def error_case_comparisons(rows):
    out=[];matched=[]
    configurations=[('T2_C1',2019,'FP','TP'),('T4_C1',2021,'FN','TN')]
    for tag,year,left,right in configurations:
        for seed in (900000,900001,900002):
            part=[r for r in rows if r['tag']==tag and int(r['seed'])==seed
                  and r['split']=='VALIDATION' and int(r['routed'])==1]
            # Compare expert decision strata, not Stage-I cases.
            left_rows=[r for r in part if int(r['label'])==(0 if left=='FP' else 1)
                       and int(r['C43_prediction'])==(1 if left=='FP' else 0)]
            right_rows=[r for r in part if int(r['label'])==(1 if right=='TP' else 0)
                        and int(r['C43_prediction'])==(1 if right=='TP' else 0)]
            for metric in METRICS:
                l=np.asarray([num(r,metric) for r in left_rows],float)
                r=np.asarray([num(r,metric) for r in right_rows],float)
                l=l[np.isfinite(l)];r=r[np.isfinite(r)]
                p=float(mannwhitneyu(l,r,alternative='two-sided').pvalue) if len(l)>=3 and len(r)>=3 else None
                out.append({'tag':tag,'year':year,'seed':seed,'comparison':left+'_vs_'+right,
                    'metric':metric,'left_n':len(l),'right_n':len(r),
                    'left_mean':float(l.mean()) if len(l) else None,
                    'right_mean':float(r.mean()) if len(r) else None,
                    'difference':float(l.mean()-r.mean()) if len(l) and len(r) else None,
                    'mann_whitney_p_descriptive':p,
                    'independent_unit':'APK; seeds repeated, not independent',
                    'selection':'frozen routed validation APKs; no matching chosen here'})
            matched.append({'tag':tag,'year':year,'seed':seed,'comparison':left+'_vs_'+right,
                'left_n':len(left_rows),'right_n':len(right_rows),
                'matched_pairs':0,'reason':'no pre-existing matched pairs for this cohort'})
    for tag in ('T2_C1','T4_C1'):
        for seed in (900000,900001,900002):
            sub=[r for r in out if r['tag']==tag and r['seed']==seed]
            adjusted=bh([r['mann_whitney_p_descriptive'] for r in sub])
            for r,q in zip(sub,adjusted):r['BH_q_within_tag_seed']=q
    return out,matched


def main():
    p=argparse.ArgumentParser();p.add_argument('--error-association',action='store_true',required=True);p.parse_args()
    rows=read_csv(OUT/'per_apk_background_responses.csv')
    pair=pair_2020(rows);intervals=matched_interval(pair)
    others,unmatched=error_case_comparisons(rows)
    write_csv(OUT/'matched_fn_tn_background.csv',pair)
    write_csv(OUT/'paired_intervals.csv',intervals)
    write_csv(OUT/'matched_fp_tp_background.csv',unmatched)
    write_csv(OUT/'background_error_association.csv',others)
    write_json(OUT/'error_phase.json',{'status':'PASS','matched_2020_pairs':22,
        'seed_rows':len(pair),'other_comparisons':len(others),
        'year_2019_2021':'development diagnostics without new pair selection',
        'future_labels_used_for_model':False})
    print('PROTOTYPE_ERROR_ASSOCIATION_DONE',len(pair),len(others),flush=True)


if __name__=='__main__':main()
