"""Evaluate frozen-route paired contrasts across T2–T4 after every exact M0 replay gate passes."""
from __future__ import annotations
from collections import defaultdict
import math
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,f1_score,accuracy_score,confusion_matrix
from common import *

TAGS=('T2_C1','T3_C1','T4_C1');SEEDS=(900000,900001,900002);ARMS=('M0','M1','M2','M3','M4','M5')

def auc(y,p):return float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None
def measure(rows,arm):
    y=np.array([int(r['label']) for r in rows]);s=np.array([int(r['stage1_prediction']) for r in rows]);route=np.array([int(r['routed']) for r in rows],bool);p=np.array([r['score'][arm] if r['routed'] else float(r['stage1_probability']) for r in rows]);v=(p>=.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,v,labels=[0,1]).ravel();basewrong=s!=y;fixed=int(np.sum(route&basewrong&(v==y)));broken=int(np.sum(route&~basewrong&(v!=y)))
    rb=[r for r in rows if r['routed']];ry=np.array([int(r['label']) for r in rb]);rp=np.array([r['score'][arm] for r in rb]);rs=np.array([int(r['stage1_prediction']) for r in rb]);maskb=rs==0;maskm=rs==1
    return dict(n=len(rows),routed=int(route.sum()),BA=float(balanced_accuracy_score(y,v)),AUC=auc(y,p),F1=float(f1_score(y,v)),accuracy=float(accuracy_score(y,v)),TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp),fixed=fixed,broken=broken,net=fixed-broken,fixed_FN=int(np.sum(route&(s==0)&(y==1)&(v==y))),fixed_FP=int(np.sum(route&(s==1)&(y==0)&(v==y))),broken_TN=int(np.sum(route&(s==0)&(y==0)&(v!=y))),broken_TP=int(np.sum(route&(s==1)&(y==1)&(v!=y))),routed_BA=float(balanced_accuracy_score(ry,(rp>=.5).astype(int))),routed_AUC=auc(ry,rp),damage_rate=broken/max(1,int(np.sum(route&~basewrong))),FN_vs_TN_AUC=auc(ry[maskb],rp[maskb]),TP_vs_FP_AUC=auc(ry[maskm],rp[maskm]),predicted_benign_n=int(maskb.sum()),predicted_malware_n=int(maskm.sum()),FN_conditional_n=int(np.sum(maskb&(ry==1))),TN_conditional_n=int(np.sum(maskb&(ry==0))),TP_conditional_n=int(np.sum(maskm&(ry==1))),FP_conditional_n=int(np.sum(maskm&(ry==0))))

def main():
    historic=read_csv(OUTPUT/'expert_capability.csv');lookup={(r['tag'],r['sha256']):r for r in historic}
    all_rows=[];cost=[]
    for tag in TAGS:
        seed_predictions={}
        for seed in SEEDS:
            d=OUTPUT/'full'/tag/str(seed)
            if read_json(d/'M0_exact_retrain_check.json')['status']!='PASS':raise RuntimeError('M0 exact gate failed')
            for arm in ARMS:
                with np.load(d/(arm+'_predictions.npz'),allow_pickle=False) as f:ids=list(f['sha256']);p=np.asarray(f['probability'])
                if len(ids)!=len(p) or len(ids)!=len(set(ids)):raise RuntimeError('prediction identifier mismatch')
                seed_predictions[(seed,arm)]=dict(zip(ids,p))
                audit=read_json(d/(arm+'_audit.json'))
                cost.append(dict(tag=tag,seed=seed,arm=arm,train_seconds=audit['seconds'],epochs=len(audit['train_checks']),optimization_steps=sum(x['steps'] for x in audit['train_checks']),valid_anchors=sum(x['valid_anchors'] for x in audit['train_checks']),positive_pairs=sum(x['positive_pairs'] for x in audit['train_checks']),model_parameters=audit['params_model'],auxiliary_parameters=0 if arm=='M0' else audit['params_projection'],background_fit_seconds='already fit in original saved checkpoint; not isolated',all_APK_inference_seconds='not separately timed in new training; full-APK prediction computed',true_top25_cost='training: same 25% fixed route; APK extraction cache reused'))
        ids=list(seed_predictions[(SEEDS[0],'M0')])
        for sid in ids:
            r=dict(lookup[(tag,sid)]);r['score']={arm:float(np.mean([seed_predictions[(seed,arm)][sid] for seed in SEEDS])) for arm in ARMS};all_rows.append(r)
    metrics=[]
    for tag in TAGS:
        rows=[r for r in all_rows if r['tag']==tag]
        for arm in ARMS:metrics.append(dict(scope=tag,arm=arm,**measure(rows,arm)))
    for arm in ARMS:metrics.append(dict(scope='pooled_T2_T4',arm=arm,**measure(all_rows,arm)))
    write_csv(OUTPUT/'model_comparisons.csv',metrics);write_csv(OUTPUT/'runtime_cost.csv',cost)
    rmap={(r['scope'],r['arm']):r for r in metrics};contrasts=[];pairs=[('M1','M0'),('M2','M1'),('M3','M2'),('M3','M4'),('M3','M5')]
    rng=np.random.RandomState(20261015);intervals=[];pvals=[]
    for a,b in pairs:
        for scope in [*TAGS,'pooled_T2_T4']:
            d=rmap[(scope,a)];ref=rmap[(scope,b)];contrasts.append(dict(scope=scope,candidate=a,reference=b,BA_gain_pp=100*(d['BA']-ref['BA']),net_gain=d['net']-ref['net'],fixed_gain=d['fixed']-ref['fixed'],broken_change=d['broken']-ref['broken'],FN_AUC_gain=(d['FN_vs_TN_AUC']-ref['FN_vs_TN_AUC']) if d['FN_vs_TN_AUC'] is not None and ref['FN_vs_TN_AUC'] is not None else None,TP_AUC_gain=(d['TP_vs_FP_AUC']-ref['TP_vs_FP_AUC']) if d['TP_vs_FP_AUC'] is not None and ref['TP_vs_FP_AUC'] is not None else None))
        # Bootstrap the same APKs, stratified by validation year and class.
        strata=defaultdict(list)
        for i,r in enumerate(all_rows):strata[(r['year'],r['label'])].append(i)
        deltas=[];nets=[]
        for _ in range(1000):
            idx=np.concatenate([rng.choice(v,size=len(v),replace=True) for v in strata.values()]);sample=[all_rows[int(i)] for i in idx];ma=measure(sample,a);mb=measure(sample,b);deltas.append(100*(ma['BA']-mb['BA']));nets.append(ma['net']-mb['net'])
        y=np.array([int(r['label']) for r in all_rows]);route=np.array([int(r['routed']) for r in all_rows],bool)
        def correct(arm):
            p=np.array([r['score'][arm] if r['routed'] else float(r['stage1_probability']) for r in all_rows]);return (p>=.5)==y
        ca=correct(a);cb=correct(b);only_a=int(np.sum(route&ca&~cb));only_b=int(np.sum(route&~ca&cb));p=float(binomtest(min(only_a,only_b),only_a+only_b,.5).pvalue) if only_a+only_b else 1.0
        intervals.append(dict(candidate=a,reference=b,scope='pooled_T2_T4',BA_gain_pp=rmap[('pooled_T2_T4',a)]['BA']*100-rmap[('pooled_T2_T4',b)]['BA']*100,BA_CI_low=float(np.quantile(deltas,.025)),BA_CI_high=float(np.quantile(deltas,.975)),net_gain=rmap[('pooled_T2_T4',a)]['net']-rmap[('pooled_T2_T4',b)]['net'],net_CI_low=float(np.quantile(nets,.025)),net_CI_high=float(np.quantile(nets,.975)),candidate_only_correct=only_a,reference_only_correct=only_b,mcnemar_exact_p=p,bootstrap_repeats=1000,CI_limitation='paired APK bootstrap does not account for overlapping training windows, family dependencies or prior development selection'))
        pvals.append(p)
    # Holm adjustment across the five pre-registered primary contrasts.
    ranked=sorted(range(len(pvals)),key=lambda i:pvals[i]);adjusted=[1.]*len(pvals);running=0
    for rank,i in enumerate(ranked):
        running=max(running,min(1,pvals[i]*(len(pvals)-rank)));adjusted[i]=running
    for r,q in zip(intervals,adjusted):r['holm_adjusted_p']=q
    write_csv(OUTPUT/'ablation_results.csv',contrasts);write_csv(OUTPUT/'paired_intervals.csv',intervals)
    def improvement(a,b):return rmap[('pooled_T2_T4',a)]['BA']-rmap[('pooled_T2_T4',b)]['BA']
    m1=improvement('M1','M0')>=.005 and rmap[('pooled_T2_T4','M1')]['net']-rmap[('pooled_T2_T4','M0')]['net']>=6
    m2=improvement('M2','M1')>0
    m3=improvement('M3','M2')>0 and improvement('M3','M4')>0 and improvement('M3','M5')>0
    positive_years=sum(rmap[(tag,'M3')]['BA']>rmap[(tag,'M0')]['BA'] for tag in TAGS)
    new_side=rmap[('pooled_T2_T4','M3')];old_side=rmap[('pooled_T2_T4','M0')]
    no_side_collapse=all(new_side[k] is not None and old_side[k] is not None and new_side[k]>=old_side[k]-.03 for k in ['FN_vs_TN_AUC','TP_vs_FP_AUC'])
    candidate=m1 and m2 and m3 and positive_years>=2 and no_side_collapse and rmap[('pooled_T2_T4','M3')]['net']>=rmap[('pooled_T2_T4','M0')]['net']+6 and all(r['holm_adjusted_p']<.05 for r in intervals if (r['candidate'],r['reference']) in [('M3','M2'),('M3','M4'),('M3','M5')])
    status='BACKGROUND_CONDITIONING_SUPPORTED_ON_DEVELOPMENT_ONLY' if candidate else 'NO_RELIABLE_INCREMENT_IDENTIFIED'
    write_json(OUTPUT/'model_selection_decision.json',dict(status=status,M1_gate=m1,M2_increment=m2,M3_increment_over_M2_M4_M5=m3,M3_positive_years=positive_years,M3_no_conditional_side_collapse=no_side_collapse,original_three_backgrounds_preserved=True,formal_mainline_changed=False,external_MalEval2026_used=False,independent_final_test=False))
    print('ETBC_SUMMARY_DONE',status,[(a,rmap[('pooled_T2_T4',a)]['BA'],rmap[('pooled_T2_T4',a)]['net']) for a in ARMS],flush=True)

if __name__=='__main__':main()
