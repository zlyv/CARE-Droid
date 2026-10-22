#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
from common import (CONTEXT_FAMILIES,DIRECT_SENSITIVE,FAMILIES,FORMAL,INVENTORY,OUTPUT,P31,binary_auc,dump_json,norm,read_csv,spearman,write_csv)

def model_prob(xtr,ytr,xva):
    from sklearn.linear_model import LogisticRegression
    if len(np.unique(ytr))<2:return np.full(len(xva),(ytr.sum()+1)/(len(ytr)+2))
    m=LogisticRegression(C=.1,solver="liblinear",max_iter=2000,class_weight="balanced",random_state=20261009);m.fit(xtr,ytr);return m.predict_proba(xva)[:,1]
def score_variant(code,indices,active,weight,y,fold):
    out=np.zeros(len(y));freq=np.zeros(len(y))
    for f in range(1,6):
        tr=fold!=f;va=fold==f;ben=tr&(y==0);x=code[:,indices] if indices else np.zeros((len(y),1));bg=(active[ben].sum(0)+1)/(ben.sum()+2);q=np.zeros((va.sum(),31))
        for r in range(31):q[:,r]=model_prob(x[ben],active[ben,r],x[va])
        support=np.maximum((q-bg)/(1-bg+1e-8),0);out[va]=(weight[va]*support).sum(1);freq[va]=(weight[va]*bg).sum(1)
    return out,freq
def auc(y,s):
    v=binary_auc(y,s);return None if not np.isfinite(v) else float(v)
def bootstrap_ci(y,s,fold,reps=4000):
    rng=np.random.default_rng(20261009);vals=[];idx=np.arange(len(y));groups=[idx[fold==f] for f in sorted(set(fold.tolist()))]
    for _ in range(reps):
        take=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups if len(g)]);v=binary_auc(y[take],s[take])
        if np.isfinite(v):vals.append(v)
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]
def main():
    formal=sorted(read_csv(FORMAL),key=lambda r:norm(r["sha256"]));invmap={norm(r["sha256"]):r for r in read_csv(INVENTORY)};sha=np.asarray([norm(r["sha256"]) for r in formal]);rows=[invmap[s] for s in sha]
    y=np.asarray([int(float(r["label"])) for r in formal]);fold=np.asarray([int(float(r["fold"])) for r in formal]);base=np.asarray([float(r["Global24__prob"]) for r in formal]);route=np.asarray([int(float(r["A25__routed"])) for r in formal],bool)
    with np.load(P31,allow_pickle=False) as a:ids=np.asarray([norm(x) for x in a["sha256"]]);pv=np.asarray(a["role_prevalence"],float)
    pos={s:i for i,s in enumerate(ids)};roles=pv[[pos[s] for s in sha]];active=(roles>0).astype(int);weight=roles/np.maximum(roles.sum(1,keepdims=True),1e-12)
    code=np.asarray([[np.log1p(float(r[k]))/max(np.log1p(float(r["invoke_count"])),1.0) for k in FAMILIES] for r in rows]);all_idx=list(range(len(FAMILIES)));ctx_idx=[FAMILIES.index(x) for x in CONTEXT_FAMILIES]
    all_score,role_freq=score_variant(code,all_idx,active,weight,y,fold);ctx_score,_=score_variant(code,ctx_idx,active,weight,y,fold)
    pm=route&(base>=.5);fp=(y[pm]==0).astype(int);pm_fold=fold[pm];all_auc=auc(fp,all_score[pm]);ctx_auc=auc(fp,ctx_score[pm]);freq_auc=auc(fp,role_freq[pm])
    fold_rows=[]
    for f in range(1,6):
        z=pm&(fold==f);fold_rows.append({"fold":f,"n":int(z.sum()),"FP":int(np.sum(y[z]==0)),"TP":int(np.sum(y[z]==1)),"all_code_auc":auc((y[z]==0).astype(int),all_score[z]),"context_only_auc":auc((y[z]==0).astype(int),ctx_score[z]),"role_frequency_auc":auc((y[z]==0).astype(int),role_freq[z])})
    write_csv(OUTPUT/"fold_results.csv",fold_rows)
    loo=[]
    for family in FAMILIES:
        indices=[i for i in all_idx if i!=FAMILIES.index(family)];s,_=score_variant(code,indices,active,weight,y,fold);v=auc(fp,s[pm]);loo.append({"base_variant":"ALL_CODE","removed_family":family,"direct_sensitive_overlap":int(family in DIRECT_SENSITIVE),"FP_vs_TP_AUC":v,"delta_from_base":None if all_auc is None or v is None else v-all_auc})
    for family in CONTEXT_FAMILIES:
        indices=[i for i in ctx_idx if i!=FAMILIES.index(family)];s,_=score_variant(code,indices,active,weight,y,fold);v=auc(fp,s[pm]);loo.append({"base_variant":"CONTEXT_ONLY","removed_family":family,"direct_sensitive_overlap":0,"FP_vs_TP_AUC":v,"delta_from_base":None if ctx_auc is None or v is None else v-ctx_auc})
    write_csv(OUTPUT/"leave_one_family_out.csv",loo)
    rng=np.random.default_rng(20261009);null=[]
    for _ in range(1000):
        z=ctx_score.copy()
        for f in range(1,6):
            ix=np.flatnonzero(pm&(fold==f));z[ix]=z[rng.permutation(ix)]
        null.append(auc(fp,z[pm]))
    valid=np.asarray([v for v in null if v is not None]);shuffle95=float(np.quantile(valid,.95))
    correlations={}
    for name,values in {"invoke_count":[float(r["invoke_count"]) for r in rows],"active_family_count":[float(r["active_family_count"]) for r in rows],"active_role_count":active.sum(1),"p31_mass":roles.sum(1)}.items():correlations[name]=spearman(ctx_score,values)
    eval_ctx=[r["context_only_auc"] for r in fold_rows if r["context_only_auc"] is not None];max_drop=max([0.0]+[-float(r["delta_from_base"]) for r in loo if r["base_variant"]=="CONTEXT_ONLY" and r["delta_from_base"] is not None])
    criteria={"context_auc>=0.65":ctx_auc is not None and ctx_auc>=.65,"context>shuffle95+0.03":ctx_auc is not None and ctx_auc>shuffle95+.03,"context>role_frequency+0.05":ctx_auc is not None and freq_auc is not None and ctx_auc>freq_auc+.05,"majority_evaluable_folds>0.5":sum(v>.5 for v in eval_ctx)>=max(2,(len(eval_ctx)+1)//2),"max_abs_size_correlation<0.70":max(abs(v or 0) for v in correlations.values())<.70,"no_single_family_drop>=0.10":max_drop<.10}
    status="CODE_CONTEXT_BENIGN_SIGNAL_SURVIVES_CONTROLS" if all(criteria.values()) else "CODE_CONTEXT_BENIGN_SIGNAL_NOT_CONFIRMED"
    result={"status":status,"all_code_auc":all_auc,"context_only_auc":ctx_auc,"role_frequency_auc":freq_auc,"context_bootstrap_95ci":bootstrap_ci(fp,ctx_score[pm],pm_fold),"matched_shuffle_95pct":shuffle95,"correlations":correlations,"max_leave_one_family_out_drop":max_drop,"criteria":criteria,"context_families":list(CONTEXT_FAMILIES),"excluded_direct_sensitive_families":sorted(DIRECT_SENSITIVE),"future_rows_loaded":0}
    dump_json(OUTPUT/"audit_results.json",result);dump_json(OUTPUT/"final_decision.json",{"status":status,"continue_to_correction_integration":all(criteria.values()),"reason":"Correction integration remains forbidden unless context-only code evidence survives all frozen controls.","future_rows_loaded":0})
    print("CODE_FUNCTION_BENIGN_AUDIT_DONE",flush=True);print("ALL_CODE_AUC="+str(all_auc)+" CONTEXT_ONLY_AUC="+str(ctx_auc)+" ROLE_FREQUENCY_AUC="+str(freq_auc),flush=True);print("CODE_FUNCTION_BENIGN_DECISION="+status,flush=True)
if __name__=="__main__":main()
