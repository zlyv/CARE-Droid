#!/usr/bin/env python3
"""LFC-V1 2017 fixed-fold RF OOF and frozen C43 complementarity. Inputs: read-only LiteStage shards/OOF and HDBC H0; outputs only LFC_OUT. Trains new first-stage RFs, never C43."""
from __future__ import annotations
import sys,time
import numpy as np
from scipy.stats import binomtest
from core import *
sys.path.insert(0,str(ROOT/'scripts/care-p31-stage1-rf-selection-source-v1'))
from stage1_core import fit_calibrated_member

SEED=20261018
def pair_interval(y,a,b,repeat=1000):
    rng=np.random.RandomState(20261018);values=[]
    for _ in range(repeat):
        ix=rng.randint(len(y),size=len(y));yy=y[ix]
        if len(set(yy))<2:continue
        values.append(lc.metrics(yy,a[ix])['BA']-lc.metrics(yy,b[ix])['BA'])
    return np.quantile(values,[.025,.975]) if values else [np.nan,np.nan]
def main():
    ids,y,fold,formal,expert=source_index();data=shard('source');legacy=legacy_source()
    if set(ids)!=set(data):raise RuntimeError('source shard index')
    p={}
    for name,old in [('ManifestOnly','L1_MANIFEST'),('BestLiteStage','LITE_NESTED_SELECTED')]:
        p[name]=np.array([legacy[(old,s)] for s in ids],float)
    p['OriginalP31RF']=np.array([float(formal[s]['p_P31__RF']) for s in ids])
    if lc.metrics(y,p['BestLiteStage'])['errors']!=52 or lc.metrics(y,p['OriginalP31RF'])['errors']!=58:raise RuntimeError('strong baselines changed')
    timing=[];top=[]
    for arm in NEW:
        q=np.full(799,np.nan)
        for f in range(1,6):
            tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f)
            selection=role_select([data[ids[i]] for i in tr],y[tr]) if arm=='RefTop16' else None
            x=feature_matrix(data,ids[tr],arm,selection);z=feature_matrix(data,ids[te],arm,selection)
            t=time.perf_counter();q[te]=fit_calibrated_member('RF',x,y[tr],z,SEED+f,inner_folds=3,jobs=2)
            timing.append(dict(arm=arm,fold=f,train_n=len(tr),test_n=len(te),width=x.shape[1],fit_predict_s=time.perf_counter()-t))
            if selection is not None:top.append(dict(fold=f,arm=arm,roles=','.join(map(str,selection))))
            print('LFC_SOURCE_ARM_FOLD_DONE',arm,f,flush=True)
        if not np.isfinite(q).all():raise RuntimeError('incomplete OOF '+arm)
        p[arm]=q
    put_csv('first_stage_training_cost.csv',timing);put_csv('train_fold_role_subsets.csv',top)
    ep=np.array([float(expert[s]['expert_probability']) for s in ids])
    all_arms=list(MAIN)+list(EXTRA)
    stage=[];preds=[];route_rows=[];fixed=[];per=[];complement=[];bins=[]
    cut=[0.,.1,.3,.5,.7,.9,1.000001]
    for arm in all_arms:
        q=p[arm];base=lc.metrics(y,q);stage.append(dict(candidate=arm,protocol='2017 fixed outer OOF; reused' if arm not in NEW else '2017 fixed outer OOF; new calibrated RF',feature_class=cost_class(arm),width=len(feature(data[ids[0]],arm,role_select([data[ids[i]] for i in np.r_[np.flatnonzero(y==0)[:20],np.flatnonzero(y==1)[:20]]],y[np.r_[np.flatnonzero(y==0)[:20],np.flatnonzero(y==1)[:20]]]) if arm=='RefTop16' else None)) if arm in NEW or arm=='ManifestOnly' else (31 if arm=='OriginalP31RF' else 'outer-selected'),**base))
        for i,s in enumerate(ids):preds.append(dict(sha256=s,fold=int(fold[i]),label=int(y[i]),candidate=arm,probability=float(q[i])))
        before=(q>=.5)==y;after=(ep>=.5)==y
        complement.append(dict(candidate=arm,n=799,both_correct=int((before&after).sum()),stage1_wrong_C43_correct=int((~before&after).sum()),stage1_correct_C43_wrong=int((before&~after).sum()),both_wrong=int((~before&~after).sum()),C43_correct_all_APK=int(after.sum())))
        for budget in (.1,.25,.4):
            route=np.zeros(799,bool)
            for f in range(1,6):
                ix=np.flatnonzero(fold==f);route[ix]=lc.route_by_score(ids[ix],1-2*np.abs(q[ix]-.5),budget)
            m,final=lc.cascade(y,q,ep,route)
            if m['stage1_errors']-m['fixed']+m['broken']!=m['errors']:raise RuntimeError('fixed/broken identity')
            rr=dict(candidate=arm,budget=budget,router='stage1_uncertainty_fold_relative',**m)
            route_rows.append(rr)
            if budget==.25:
                fixed.append(rr)
                for i,s in enumerate(ids):
                    fixed_one=int((not before[i]) and ((final[i]>=.5)==y[i]));broken_one=int(before[i] and ((final[i]>=.5)!=y[i]))
                    per.append(dict(sha256=s,fold=int(fold[i]),label=int(y[i]),candidate=arm,stage1_probability=float(q[i]),C43_probability=float(ep[i]),routed=int(route[i]),final_probability=float(final[i]),stage1_correct=int(before[i]),C43_correct=int(after[i]),fixed=fixed_one,broken=broken_one))
                for lo,hi in zip(cut[:-1],cut[1:]):
                    ix=(q>=lo)&(q<hi)
                    bins.append(dict(candidate=arm,probability_low=lo,probability_high=min(hi,1.),n=int(ix.sum()),stage1_errors=int((~before&ix).sum()),C43_correct_stage1_errors=int((~before&after&ix).sum()),routed=int((route&ix).sum()),fixed=int((~before&((final>=.5)==y)&ix).sum()),broken=int((before&((final>=.5)!=y)&ix).sum())))
    put_csv('first_stage_oof_predictions.csv',preds);put_csv('first_stage_metrics.csv',stage)
    put_csv('stage1_stage2_complementarity.csv',complement);put_csv('routing_budget_results.csv',route_rows)
    put_csv('fixed_broken_by_candidate.csv',fixed);put_csv('source_25pct_per_apk.csv',per);put_csv('confidence_bin_analysis.csv',bins)
    summary=[]
    ref=p['BestLiteStage'];refc=(ref>=.5)==y
    for arm in all_arms:
        a=p[arm];ac=(a>=.5)==y;w=int((ac&~refc).sum());l=int((~ac&refc).sum());ci=pair_interval(y,a,ref)
        summary.append(dict(comparison=arm+' vs BestLiteStage Stage-I',BA_difference=lc.metrics(y,a)['BA']-lc.metrics(y,ref)['BA'],CI_low=float(ci[0]),CI_high=float(ci[1]),candidate_only_correct=w,reference_only_correct=l,mcnemar_exact_p=float(binomtest(min(w,l),w+l,.5).pvalue) if w+l else 1.))
    put_csv('paired_statistics.csv',summary)
    print('LFC_SOURCE_DONE candidates',len(all_arms),'C43_predictions',len(ep),flush=True)
if __name__=='__main__':main()
