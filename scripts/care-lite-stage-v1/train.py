#!/usr/bin/env python3
"""LiteStage-V1 source-only nested RF and matched frozen Direct-C43 OOF comparison. Inputs: new source shards, formal P31 predictions, SCBE/HDBC OOF expert. Writes only new LiteStage results; trains diagnostic RF/committee on 2017 only; no APK reads."""
from __future__ import annotations
import json,sys,time
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from scipy.stats import binomtest
from common import *

sys.path.insert(0,str(ROOT/'scripts/care-p31-stage1-rf-selection-source-v1'))
from stage1_core import fit_calibrated_member,fit_calibrated_committee,mean_opposition

ARMS=['L1_MANIFEST','L2_DEXMETA']+[f'L3_{x}' for x in ('COUNT','PRESENCE','LOG','TOP8','TOP16','TOP24')]+[f'L4_{x}' for x in ('COUNT','PRESENCE','LOG','TOP8','TOP16','TOP24')]
SEED=20261017
def records_for(ids,allrows):return [allrows[s] for s in ids]
def design(fitrows,testrows,yfit,arm):
    role=choose_roles(fitrows,yfit,arm)
    return matrix(fitrows,arm,role),matrix(testrows,arm,role),role
def lite_rf(x,y,z,seed,trees=300):
    m=RandomForestClassifier(n_estimators=trees,max_features='sqrt',min_samples_leaf=2,class_weight='balanced',n_jobs=2,random_state=seed)
    m.fit(x,y)
    return m.predict_proba(z)[:,list(m.classes_).index(1)]
def nested_score(rows,y,arm,seed):
    split=StratifiedKFold(3,shuffle=True,random_state=seed);score=[]
    for k,(it,iv) in enumerate(split.split(np.zeros(len(y)),y),1):
        fit=[rows[j] for j in it];val=[rows[j] for j in iv];x,z,_=design(fit,val,y[it],arm)
        p=lite_rf(x,y[it],z,seed+100*k,100);score.append(metrics(y[iv],p)['BA'])
    return float(np.mean(score)),score
def per_apk_method(ids,y,b,e,route,name,fold):
    _,final=cascade(y,b,e,route);out=[]
    for i,sid in enumerate(ids):
        before=(b[i]>=.5)==y[i];after=(final[i]>=.5)==y[i]
        out.append(dict(method=name,sha256=sid,fold=int(fold),label=int(y[i]),stage1_probability=float(b[i]),routed=int(route[i]),expert_probability=float(e[i]),final_probability=float(final[i]),fixed=int(not before and after),broken=int(before and not after)))
    return out
def interval(y,a,b,repeats=1000,seed=216):
    rng=np.random.RandomState(seed);n=len(y);diff=[]
    for _ in range(repeats):
        ix=rng.randint(n,size=n);yy=y[ix]
        if len(set(yy))<2:continue
        diff.append(metrics(yy,a[ix])['BA']-metrics(yy,b[ix])['BA'])
    return np.quantile(diff,[.025,.975]) if diff else [np.nan,np.nan]

def main():
    lineage,original=source_contract();exp=frozen_expert_source();shards,missing=load_shards('source')
    if missing:
        writejson('source_extraction_blocker.json',dict(status='BLOCKED',missing=len(missing),example=missing[:20],policy='no silent exclusion from 799 formal comparison'))
        raise RuntimeError(f'source shards missing {len(missing)}; no 799 matched comparison')
    ids=np.array(sorted(shards));y=np.array([int(original[s]['label']) for s in ids]);fold=np.array([int(original[s]['fold']) for s in ids]);expert=np.array([float(exp[s]['expert_probability']) for s in ids]);rf=np.array([float(original[s]['p_P31__RF']) for s in ids]);svm=np.array([float(original[s]['p_P31__SVM']) for s in ids]);member=np.array([[float(original[s][f'p_P31__{m}']) for m in ('LR','RF','XGB','SVM')] for s in ids])
    if metrics(y,rf)['errors']!=58 or metrics(y,svm)['errors']!=97:raise RuntimeError('formal Stage-I identity')
    if any(int(exp[s]['split'][1:])!=fold[i] or int(exp[s]['label'])!=y[i] for i,s in enumerate(ids)):raise RuntimeError('C43 OOF Fold/label')
    oof={arm:np.full(799,np.nan) for arm in ARMS};selected=np.full(799,np.nan);committee=np.full((799,4),np.nan);selected_names={};nested_rows=[];source_rows=[];timing=[]
    for outer in range(1,6):
        tr=np.flatnonzero(fold!=outer);va=np.flatnonzero(fold==outer)
        fitrows=records_for(ids[tr],shards);valrows=records_for(ids[va],shards);yt=y[tr]
        if len(tr)+len(va)!=799 or set(ids[tr])&set(ids[va]):raise RuntimeError('outer leak')
        ranking=[]
        for arm in ARMS:
            t=time.perf_counter();value,parts=nested_score(fitrows,yt,arm,SEED+outer)
            ranking.append((value,arm));nested_rows.append(dict(outer_fold=outer,arm=arm,inner_BA=value,inner_1=parts[0],inner_2=parts[1],inner_3=parts[2],selection_trees=100,seconds=time.perf_counter()-t))
        winner=sorted(ranking,key=lambda q:(-q[0],q[1]))[0][1];selected_names[outer]=winner
        print('LITESTAGE_OUTER_SELECTED',outer,winner,flush=True)
        for arm in ARMS:
            t=time.perf_counter();x,z,roles=design(fitrows,valrows,yt,arm)
            p=fit_calibrated_member('RF',x,yt,z,SEED+outer,inner_folds=3,jobs=2)
            oof[arm][va]=p;timing.append(dict(outer_fold=outer,arm=arm,fit_infer_s=time.perf_counter()-t,fit_n=len(tr),test_n=len(va),width=x.shape[1],role_selected=','.join(map(str,roles)) if roles is not None else 'all'))
        selected[va]=oof[winner][va]
        x,z,_=design(fitrows,valrows,yt,winner);committee[va]=fit_calibrated_committee(x,yt,z,SEED+outer,inner_folds=3,jobs=2)
        if np.max(np.abs(committee[va,1]-selected[va]))>1e-8:raise RuntimeError('selected RF and committee RF diverge')
        print('LITESTAGE_OUTER_DONE',outer,flush=True)
    if any(not np.isfinite(v).all() for v in [selected,committee,*oof.values()]):raise RuntimeError('incomplete OOF')
    writecsv('nested_model_selection.csv',nested_rows);writecsv('stage1_fit_inference_cost.csv',timing)
    for arm,p in oof.items():
        for i,sid in enumerate(ids):source_rows.append(dict(sha256=sid,fold=int(fold[i]),label=int(y[i]),arm=arm,probability=float(p[i]),selected_for_fold=int(arm==selected_names[fold[i]])))
    for arm,p in [('L5_P31_RF',rf),('P31_SVM_AUX',svm),('LITE_NESTED_SELECTED',selected)]:
        for i,sid in enumerate(ids):source_rows.append(dict(sha256=sid,fold=int(fold[i]),label=int(y[i]),arm=arm,probability=float(p[i]),selected_for_fold=int(arm=='LITE_NESTED_SELECTED')))
    writecsv('stage1_oof_predictions.csv',source_rows)
    arms_metric=[dict(arm=arm,protocol='calibrated RF fixed outer OOF, arm exploratory',width=next(t['width'] for t in timing if t['arm']==arm),**metrics(y,p)) for arm,p in oof.items()]
    arms_metric.extend([dict(arm='L5_P31_RF',protocol='frozen historical calibrated OOF',width=31,**metrics(y,rf)),dict(arm='P31_SVM_AUX',protocol='frozen historical calibrated OOF',width=31,**metrics(y,svm)),dict(arm='LITE_NESTED_SELECTED',protocol='outer-train inner-selected calibrated RF OOF',width='varies by fold',**metrics(y,selected))])
    writecsv('stage1_source2017_metrics.csv',arms_metric)
    writecsv('role_compression_results.csv',[r for r in arms_metric if r['arm'].startswith(('L3_','L4_'))])
    score_lite=mean_opposition(committee,selected);score_rf=mean_opposition(member,rf)
    route_lite=np.zeros(799,bool);route_rf=np.zeros(799,bool);uncertainty={k:np.zeros(799,bool) for k in ('L1_MANIFEST','L2_DEXMETA','L3_COUNT','L4_COUNT')}
    budget_rows=[]
    for budget in (.1,.25,.4):
        route_by_budget=np.zeros(799,bool);rf_by_budget=np.zeros(799,bool)
        for outer in range(1,6):
            ix=np.flatnonzero(fold==outer);route_by_budget[ix]=route_by_score(ids[ix],score_lite[ix],budget);rf_by_budget[ix]=route_by_score(ids[ix],score_rf[ix],budget)
            if budget==.25:
                route_lite[ix]=route_by_budget[ix];route_rf[ix]=rf_by_budget[ix]
                for arm in uncertainty:
                    uncertainty[arm][ix]=route_by_score(ids[ix],-np.abs(oof[arm][ix]-.5),.25)
        for label,b,route in [('LITE_NESTED_SELECTED',selected,route_by_budget),('L5_P31_RF',rf,rf_by_budget)]:
            m,_=cascade(y,b,expert,route);budget_rows.append(dict(stage1=label,budget=budget,router='member-calibrated MeanOpposition, batch-relative per fixed outer fold',**m))
    writecsv('router_budget_comparison.csv',budget_rows)
    methods={'T0_P31_RF':(rf,np.zeros(799,bool)),'T1_P31_RF_C43':(rf,route_rf),'T2_MANIFEST_C43':(oof['L1_MANIFEST'],uncertainty['L1_MANIFEST']),'T3_DEXMETA_C43':(oof['L2_DEXMETA'],uncertainty['L2_DEXMETA']),'T4_REF_C43':(oof['L3_COUNT'],uncertainty['L3_COUNT']),'T5_INVOKE_C43':(oof['L4_COUNT'],uncertainty['L4_COUNT']),'T6_LITE_ONLY':(selected,np.zeros(799,bool)),'T7_LITE_C43':(selected,route_lite),'SVM_HISTORICAL_C43':(svm,np.array([int(exp[s]['routed']) for s in ids],bool))}
    comp=[];per=[]
    for name,(b,route) in methods.items():
        m,_=cascade(y,b,expert,route);comp.append(dict(method=name,router='MeanOpposition' if name in ('T1_P31_RF_C43','T7_LITE_C43') else 'RF uncertainty' if name.startswith(('T2','T3','T4','T5')) else 'historical' if name=='SVM_HISTORICAL_C43' else 'none',**m))
        for outer in range(1,6):
            ix=np.flatnonzero(fold==outer);per+=per_apk_method(ids[ix],y[ix],b[ix],expert[ix],route[ix],name,outer)
    writecsv('two_stage_source2017_metrics.csv',comp);writecsv('two_stage_source2017_predictions.csv',per)
    writecsv('fixed_broken_comparison.csv',[dict(method=r['method'],stage1_errors=r['stage1_errors'],routed=r['routed'],routed_errors=r['routed_stage1_errors'],fixed=r['fixed'],broken=r['broken'],net=r['net'],BA=r['BA']) for r in comp])
    writecsv('conditional_auc_comparison.csv',[dict(method=r['method'],FN_TN_AUC=r['FN_TN_AUC'],TP_FP_AUC=r['TP_FP_AUC'],routed=r['routed']) for r in comp])
    paired=[];scoremap={r['method']:r for r in comp}
    pp={name:np.array([float(r['final_probability']) for r in sorted((q for q in per if q['method']==name),key=lambda x:x['sha256'])]) for name in methods}
    for other in ('T6_LITE_ONLY','T0_P31_RF','T1_P31_RF_C43'):
        a=pp['T7_LITE_C43'];b=pp[other];ak=(a>=.5)==y;bk=(b>=.5)==y;win=int((ak&~bk).sum());loss=int((~ak&bk).sum());ci=interval(y,a,b)
        paired.append(dict(comparison='T7_LITE_C43 vs '+other,BA_difference=metrics(y,a)['BA']-metrics(y,b)['BA'],CI_low=float(ci[0]),CI_high=float(ci[1]),candidate_only_correct=win,control_only_correct=loss,mcnemar_exact_p=float(binomtest(min(win,loss),win+loss,.5).pvalue) if win+loss else 1.))
    writecsv('paired_statistical_results.csv',paired)
    # Global final arm uses only source inner-validation scores, never future data or outer OOF ranking.
    byarm={arm:float(np.mean([r['inner_BA'] for r in nested_rows if r['arm']==arm])) for arm in ARMS};final_arm=sorted(byarm,key=lambda a:(-byarm[a],a))[0]
    writejson('source_selection.json',dict(selected_by_outer=selected_names,global_final_arm=final_arm,selection='mean of outer-train inner-CV BA, tie lexical; outer OOF only reports performance',inner_mean_BA=byarm,stage1_seed=SEED,role_selection='train-only standardized mean difference',router='MeanOpposition for selected arm; batch-relative per fold/year',C43='SCBE M0 frozen route-independent outer-trained Direct-C43'))
    print('LITESTAGE_SOURCE_DONE global_final_arm',final_arm,flush=True)
if __name__=='__main__':main()
