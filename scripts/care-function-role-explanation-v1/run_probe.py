#!/usr/bin/env python3
from __future__ import annotations
import json
from collections import Counter
import numpy as np
from common import FORMAL,OUTPUT,P31,binary_auc,dump_json,norm,read_csv,spearman,write_csv
from taxonomy import FAMILIES

def fit_predict(xtr,ytr,xva):
    from sklearn.linear_model import LogisticRegression
    if len(np.unique(ytr))<2:return np.full(len(xva),(ytr.sum()+1)/(len(ytr)+2))
    m=LogisticRegression(C=.1,solver="liblinear",max_iter=2000,class_weight="balanced",random_state=20261009);m.fit(xtr,ytr);return m.predict_proba(xva)[:,1]
def auc_or_none(y,s):
    v=binary_auc(y,s);return None if not np.isfinite(v) else float(v)
def main():
    formal=sorted(read_csv(FORMAL),key=lambda r:norm(r["sha256"]));sha=np.asarray([norm(r["sha256"]) for r in formal]);y=np.asarray([int(float(r["label"])) for r in formal]);fold=np.asarray([int(float(r["fold"])) for r in formal]);base=np.asarray([float(r["Global24__prob"]) for r in formal]);route=np.asarray([int(float(r["A25__routed"])) for r in formal],bool)
    with np.load(P31,allow_pickle=False) as a:pids=np.asarray([norm(x) for x in a["sha256"]]);pv=np.asarray(a["role_prevalence"],float)
    pos={s:i for i,s in enumerate(pids)};roles=pv[[pos[s] for s in sha]];active=(roles>0).astype(int);weight=roles/np.maximum(roles.sum(1,keepdims=True),1e-12)
    inv={norm(r["sha256"]):r for r in read_csv(OUTPUT/"function_feature_inventory.csv")};records=[inv[s] for s in sha]
    code=np.asarray([[np.log1p(float(r[k]))/max(np.log1p(float(r["invoke_count"])),1.0) for k in FAMILIES] for r in records])
    token_sets=[]
    shard=OUTPUT/"cache/function_features"
    for s in sha:token_sets.append(set(json.loads((shard/(s+".json")).read_text(encoding="utf-8"))["manifest_tokens"]))
    manifest_score=np.zeros(len(y));code_score=np.zeros(len(y));dual_score=np.zeros(len(y));unexplained=np.zeros(len(y));audits=[]
    for f in range(1,6):
        tr=fold!=f;va=fold==f;ben=tr&(y==0);counts=Counter(t for i in np.flatnonzero(ben) for t in token_sets[i]);vocab=sorted(t for t,c in counts.items() if c>=3)
        xm=np.asarray([[int(t in token_sets[i]) for t in vocab] for i in range(len(y))],float)
        if not vocab:xm=np.zeros((len(y),1))
        bg=(active[ben].sum(0)+1)/(ben.sum()+2);qm=np.zeros((va.sum(),31));qc=np.zeros_like(qm)
        for r in range(31):
            qm[:,r]=fit_predict(xm[ben],active[ben,r],xm[va]);qc[:,r]=fit_predict(code[ben],active[ben,r],code[va])
        sm=np.maximum((qm-bg)/(1-bg+1e-8),0);sc=np.maximum((qc-bg)/(1-bg+1e-8),0);dual=np.sqrt(sm*sc)
        manifest_score[va]=(weight[va]*sm).sum(1);code_score[va]=(weight[va]*sc).sum(1);dual_score[va]=(weight[va]*dual).sum(1);unexplained[va]=(weight[va]*(1-np.maximum(sm,sc))).sum(1)
        audits.append({"fold":f,"benign_train":int(ben.sum()),"manifest_vocab":len(vocab),"valid_role_targets":int(np.sum((active[ben].sum(0)>0)&(active[ben].sum(0)<ben.sum())))})
    pm=route&(base>=.5);pb=route&(base<.5);fp=(y[pm]==0).astype(int);fn=(y[pb]==1).astype(int)
    count_score=np.asarray([float(r["manifest_token_count"])+float(r["active_family_count"]) for r in records])
    pooled={"manifest_FP_vs_TP_AUC":auc_or_none(fp,manifest_score[pm]),"code_FP_vs_TP_AUC":auc_or_none(fp,code_score[pm]),"dual_FP_vs_TP_AUC":auc_or_none(fp,dual_score[pm]),"count_FP_vs_TP_AUC":auc_or_none(fp,count_score[pm]),"unexplained_FN_vs_TN_AUC":auc_or_none(fn,unexplained[pb]),"routed_predicted_malware":int(pm.sum()),"routed_predicted_benign":int(pb.sum())}
    folds=[]
    for f in range(1,6):
        m=pm&(fold==f);b=pb&(fold==f);folds.append({"fold":f,"routed":int(np.sum(route&(fold==f))),"dual_FP_vs_TP_AUC":auc_or_none((y[m]==0).astype(int),dual_score[m]),"unexplained_FN_vs_TN_AUC":auc_or_none((y[b]==1).astype(int),unexplained[b])})
    write_csv(OUTPUT/"fold_probe_results.csv",folds);write_csv(OUTPUT/"fold_model_audit.csv",audits)
    write_csv(OUTPUT/"apk_function_explanation.csv",[{"sha256":sha[i],"fold":int(fold[i]),"label":int(y[i]),"routed":int(route[i]),"p_RF":base[i],"manifest_support":manifest_score[i],"code_support":code_score[i],"dual_legitimate_support":dual_score[i],"unexplained_sensitive_mass":unexplained[i]} for i in range(len(y))])
    rng=np.random.default_rng(20261009);null=[]
    for rep in range(200):
        z=dual_score.copy()
        for f in range(1,6):
            for side in (0,1):
                ix=np.flatnonzero(route&(fold==f)&((base>=.5)==side));z[ix]=z[rng.permutation(ix)]
        null.append(auc_or_none(fp,z[pm]))
    valid=np.asarray([v for v in null if v is not None]);shuffle95=float(np.quantile(valid,.95));eval_folds=[r["dual_FP_vs_TP_AUC"] for r in folds if r["dual_FP_vs_TP_AUC"] is not None]
    corr={"manifest_token_count":spearman(dual_score,[float(r["manifest_token_count"]) for r in records]),"active_family_count":spearman(dual_score,[float(r["active_family_count"]) for r in records]),"invoke_count":spearman(dual_score,[float(r["invoke_count"]) for r in records])}
    criteria={"dual_auc>=0.65":pooled["dual_FP_vs_TP_AUC"] is not None and pooled["dual_FP_vs_TP_AUC"]>=.65,"dual>shuffle95+0.03":pooled["dual_FP_vs_TP_AUC"] is not None and pooled["dual_FP_vs_TP_AUC"]>shuffle95+.03,"majority_evaluable_folds>0.5":sum(v>.5 for v in eval_folds)>=max(2,(len(eval_folds)+1)//2),"max_size_correlation<0.70":max(abs(v or 0) for v in corr.values())<.70}
    status="FUNCTIONAL_BENIGN_EVIDENCE_SUFFICIENT" if all(criteria.values()) else "FUNCTIONAL_BENIGN_EVIDENCE_NOT_CONFIRMED"
    dump_json(OUTPUT/"probe_results.json",{"pooled":pooled,"correlations":corr,"shuffle_auc_95pct":shuffle95,"criteria":criteria,"status":status,"future_rows_loaded":0})
    dump_json(OUTPUT/"final_decision.json",{"status":status,"continue_to_correction_integration":all(criteria.values()),"reason":"Positive benign evidence must beat matched null before any bounded-correction experiment.","future_rows_loaded":0})
    print("FCBE_PROBE_DONE",flush=True);print(json.dumps(pooled,sort_keys=True),flush=True);print("FCBE_DECISION="+status,flush=True)
if __name__=="__main__":main()
