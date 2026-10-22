#!/usr/bin/env python3
"""Strict 2017 OOF evaluation of method-local semantic representations."""
from __future__ import annotations
import csv, json, time
from pathlib import Path
import numpy as np
from scipy.stats import binomtest, spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT=Path("artifacts/care-litelocal-semantic-source-v1")
ARMS=("P31","COOC16","TRANS16","FUSED63")
MEMBERS=("LR","RF","XGB","SVM")

def read(path):
    with Path(path).open("r",encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def norm(value):return str(value or "").strip().casefold()
def write(path,rows):
    if not rows:return
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields:fields.append(key)
    with path.open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
def make(name,seed,jobs=4):
    if name=="LR":return Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=1,solver="liblinear",max_iter=5000,random_state=seed))])
    if name=="RF":return RandomForestClassifier(n_estimators=300,max_features="sqrt",min_samples_leaf=2,class_weight="balanced",n_jobs=jobs,random_state=seed)
    if name=="XGB":return XGBClassifier(n_estimators=300,max_depth=4,learning_rate=.05,subsample=.8,colsample_bytree=.8,reg_lambda=1,objective="binary:logistic",eval_metric="logloss",tree_method="hist",n_jobs=jobs,random_state=seed,verbosity=0)
    if name=="SVM":return Pipeline([("s",StandardScaler()),("m",SVC(C=1,kernel="rbf",gamma="scale",probability=False,random_state=seed))])
    raise KeyError(name)
def raw(model,x):
    if hasattr(model,"decision_function"):return np.asarray(model.decision_function(x),float).reshape(-1)
    p=np.clip(model.predict_proba(x)[:,list(model.classes_).index(1)],1e-8,1-1e-8);return np.log(p/(1-p))
def arm_features(arm,prevalence,cooc,transitions,fit_idx,apply_idx,seed):
    if arm=="P31":return prevalence[fit_idx],prevalence[apply_idx]
    if arm in ("COOC16","FUSED63"):
        co=TruncatedSVD(n_components=16,n_iter=10,random_state=seed);co_fit=co.fit_transform(cooc[fit_idx]);co_apply=co.transform(cooc[apply_idx])
    if arm in ("TRANS16","FUSED63"):
        tx=TruncatedSVD(n_components=16,n_iter=10,random_state=seed+1);tx_fit=tx.fit_transform(transitions[fit_idx]);tx_apply=tx.transform(transitions[apply_idx])
    if arm=="COOC16":return co_fit,co_apply
    if arm=="TRANS16":return tx_fit,tx_apply
    if arm=="FUSED63":return np.hstack([prevalence[fit_idx],co_fit,tx_fit]),np.hstack([prevalence[apply_idx],co_apply,tx_apply])
    raise KeyError(arm)
def committee_nested(arm,prevalence,cooc,transitions,y,tr,va,seed,jobs=4):
    splits=list(StratifiedKFold(4,shuffle=True,random_state=seed).split(tr,y[tr]));out=np.zeros((len(va),4));audit=[]
    transformed=[]
    for k,(fit_pos,cal_pos) in enumerate(splits,1):
        fit_idx,cal_idx=tr[fit_pos],tr[cal_pos];transformed.append((*arm_features(arm,prevalence,cooc,transitions,fit_idx,cal_idx,seed+100*k),fit_pos,cal_pos))
    xtr,xva=arm_features(arm,prevalence,cooc,transitions,tr,va,seed+999)
    for j,name in enumerate(MEMBERS):
        inner=np.full(len(tr),np.nan)
        for k,(inner_fit,inner_cal,fit_pos,cal_pos) in enumerate(transformed,1):
            model=make(name,seed+10000*j+100*k,jobs);model.fit(inner_fit,y[tr[fit_pos]]);inner[cal_pos]=raw(model,inner_cal)
        platt=LogisticRegression(C=1e6,solver="lbfgs",max_iter=2000,random_state=seed+10000*j+999);platt.fit(inner.reshape(-1,1),y[tr])
        model=make(name,seed+10000*j,jobs);model.fit(xtr,y[tr]);out[:,j]=np.clip(platt.predict_proba(raw(model,xva).reshape(-1,1))[:,1],1e-8,1-1e-8)
        audit.append({"member":name,"dimension":xtr.shape[1],"train_n":len(tr),"validation_n":len(va),"protocol_seed":seed,"target_labels_received":False,"inner_svd_fit_only":True})
    return out,audit,xtr,xva
def top_foldwise(score,folds,ids):
    route=np.zeros(len(score),bool)
    for fold in sorted(set(folds)):
        idx=np.flatnonzero(folds==fold);n=int(round(len(idx)*.25));order=np.lexsort((ids[idx],-score[idx]));route[idx[order[:n]]]=True
    return route
def metrics(y,p):
    pred=p>=.5;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"BA":float(balanced_accuracy_score(y,pred)),"AUC":float(roc_auc_score(y,p)),"F1":float(f1_score(y,pred,zero_division=0)),"TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp),"errors":int(np.sum(pred!=y))}
def capture(error,route,prefix=""):
    return {prefix+"captured_errors":int(np.sum(error&route)),prefix+"base_errors":int(error.sum()),prefix+"error_recall":float(np.sum(error&route)/max(error.sum(),1)),prefix+"routing_precision":float(np.sum(error&route)/max(route.sum(),1))}
def size_probe(fold_data,size,folds):
    lo,hi=np.quantile(size,[.25,.75]);extreme=(size<=lo)|(size>=hi);target=(size>=hi).astype(int);pred=np.full(len(size),np.nan)
    for fold,(tr,va,xtr,xva) in fold_data.items():
        keep_tr=extreme[tr];keep_va=extreme[va]
        model=Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=1,solver="liblinear",max_iter=5000,random_state=20260925+fold))])
        model.fit(xtr[keep_tr],target[tr][keep_tr]);pred[va[keep_va]]=model.predict_proba(xva[keep_va])[:,1]
    ok=np.isfinite(pred)&extreme
    return float(roc_auc_score(target[ok],pred[ok])),int(ok.sum())

def main():
    cfg=json.loads((ROOT/"resolved_inputs.json").read_text());router=read(cfg["router"]);rm={norm(r["sha256"]):r for r in router}
    with np.load(ROOT/"local_semantic_features_2017.npz",allow_pickle=False) as z:
        ids=np.asarray([norm(x) for x in z["sha256"]]);prevalence=z["role_prevalence"].astype(float);cooc=z["within_method_cooccurrence"].astype(float);transitions=z["directed_adjacent_transitions"].astype(float);audit=z["audit_values"].astype(float);audit_names=[str(x) for x in z["audit_names"]]
    y=np.asarray([int(float(rm[s]["label"])) for s in ids]);folds=np.asarray([int(float(rm[s]["fold"])) for s in ids]);official=np.asarray([float(rm[s]["p_stage1"]) for s in ids]);official_route=np.asarray([int(float(rm[s]["route_DI"]))==1 for s in ids])
    if len(ids)!=799 or len(set(ids))!=799 or official_route.sum()!=200:raise RuntimeError("BLOCKED_BY_CONTRACT: cohort/router changed")
    probs={arm:np.full((799,4),np.nan) for arm in ARMS};fold_data={arm:{} for arm in ARMS};fit_audit=[];timing=[]
    for fold in sorted(set(folds)):
        tr=np.flatnonzero(folds!=fold);va=np.flatnonzero(folds==fold);seed=20260925+100000*fold
        for arm in ARMS:
            started=time.perf_counter();out,audit_rows,xtr,xva=committee_nested(arm,prevalence,cooc,transitions,y,tr,va,seed);probs[arm][va]=out;fold_data[arm][fold]=(tr,va,xtr,xva)
            timing.append({"arm":arm,"fold":fold,"dimension":xtr.shape[1],"seconds":time.perf_counter()-started});fit_audit.extend({"arm":arm,"fold":fold,**r} for r in audit_rows)
        print(f"LiteLocal source OOF fold {fold}/5 done",flush=True)
    official_error=(official>=.5)!=y;aggregate=[];samples=[];paired=[];routes={}
    aggregate.append({"arm":"FROZEN_GLOBAL24_DI25_REFERENCE",**metrics(y,official),"routed":200,**capture(official_error,official_route,"official_")})
    for arm in ARMS:
        p=probs[arm][:,0];own_error=(p>=.5)!=y;decision=p>=.5;score=np.where(decision[:,None],1-probs[arm],probs[arm]).mean(1);route=top_foldwise(score,folds,ids);routes[arm]=(route,score)
        candidate_only=int(np.sum(official_error&route&~official_route));di_only=int(np.sum(official_error&official_route&~route));p_exact=float(binomtest(min(candidate_only,di_only),candidate_only+di_only,.5).pvalue) if candidate_only+di_only else 1.0
        aggregate.append({"arm":arm,"dimension":{"P31":31,"COOC16":16,"TRANS16":16,"FUSED63":63}[arm],**metrics(y,p),"routed":int(route.sum()),**capture(own_error,route,"own_"),**capture(official_error,route,"official_")})
        paired.append({"candidate":arm,"candidate_only_official_errors":candidate_only,"DI_only_official_errors":di_only,"official_capture_delta":candidate_only-di_only,"exact_p":p_exact,"route_overlap":int(np.sum(route&official_route))})
    shortcut=[];corr=[]
    nuisance=("apk_bytes","class_count","method_ref_count","sensitive_method_count","mapped_invoke_count")
    for arm in ARMS:
        max_auc=0.;max_rho=0.
        for name in nuisance:
            values=audit[:,audit_names.index(name)];auc,n=size_probe(fold_data[arm],values,folds);max_auc=max(max_auc,auc);shortcut.append({"arm":arm,"audit_size_variable":name,"extreme_quartile_OOF_AUC":auc,"n":n})
        for fold,(tr,va,xtr,xva) in fold_data[arm].items():
            for j in range(xva.shape[1]):
                for name in nuisance:
                    rho=float(spearmanr(xva[:,j],audit[va,audit_names.index(name)]).statistic);rho=0. if not np.isfinite(rho) else rho;max_rho=max(max_rho,abs(rho));corr.append({"arm":arm,"fold":fold,"component":j,"audit_size_variable":name,"spearman_rho":rho,"abs_rho":abs(rho)})
        for row in aggregate:
            if row["arm"]==arm:row["max_fold_component_abs_size_spearman"]=max_rho;row["max_size_quartile_OOF_AUC"]=max_auc;row["shortcut_safe"]=int(max_rho<=.70 and max_auc<=.80)
    for i,sid in enumerate(ids):
        row={"sha256":sid,"year":2017,"fold":int(folds[i]),"label":int(y[i]),"p_frozen_Global24":official[i],"route_frozen_DI25":int(official_route[i])}
        for arm in ARMS:row["p_stage1__"+arm]=probs[arm][i,0];row["U_DI__"+arm]=routes[arm][1][i];row["route__"+arm]=int(routes[arm][0][i])
        samples.append(row)
    write(ROOT/"aggregate_metrics.csv",aggregate);write(ROOT/"paired_route_capture_vs_frozen_DI25.csv",paired);write(ROOT/"size_predictability.csv",shortcut);write(ROOT/"fold_component_size_correlations.csv",corr);write(ROOT/"fit_audit.csv",fit_audit);write(ROOT/"timing_model.csv",timing);write(ROOT/"source_oof_predictions.csv",samples)
    extraction=audit[:,audit_names.index("total_seconds")];lines=["CARE-Droid strict-2017 LiteLocalSemantic source v1","","Within-method invoke semantics only; no Analysis object, inter-method call graph, Host, RoleGraph, Ego, or future row.","All raw size/count variables are audit-only. Cooccurrence and transition SVD are fitted on each outer-training fold only.","Candidate routes are also evaluated against the same frozen Global24 error set used by frozen DI25.","","Results:"]
    for row in aggregate:
        if row["arm"].startswith("FROZEN"):
            lines.append(f"{row['arm']}: BA={row['BA']:.6f} AUC={row['AUC']:.6f} errors={row['errors']} official_capture={row['official_captured_errors']}/{row['official_base_errors']}")
        else:
            lines.append(f"{row['arm']}: BA={row['BA']:.6f} AUC={row['AUC']:.6f} F1={row['F1']:.6f} own_errors={row['errors']} own_capture={row['own_captured_errors']}/{row['own_base_errors']} official_capture={row['official_captured_errors']}/{row['official_base_errors']} shortcut_safe={row['shortcut_safe']}")
    lines += ["","Paired frozen-Global24 error capture versus frozen DI25:"]+[f"{r['candidate']}: candidate_only={r['candidate_only_official_errors']} DI_only={r['DI_only_official_errors']} delta={r['official_capture_delta']} exact_p={r['exact_p']:.6g}" for r in paired]
    lines += ["",f"Extraction timing: total={extraction.sum():.3f}s mean={extraction.mean():.6f}s median={np.median(extraction):.6f}s p90={np.quantile(extraction,.9):.6f}s max={extraction.max():.6f}s","Shortcut gate: maximum fold-component |Spearman| <= 0.70 and every extreme-size-quartile OOF AUC <= 0.80.","No candidate is authorized for future evaluation or replacement until source utility and shortcut safety are inspected."]
    (ROOT/"REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8");print("LITELOCAL_SEMANTIC_SOURCE_DONE");print("\n".join(lines))

if __name__=="__main__":main()

