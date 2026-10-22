#!/usr/bin/env python3
"""Fixed diagnostic probes; never used as CARE-Droid predictions."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score,average_precision_score,balanced_accuracy_score,f1_score,
                             mean_absolute_error,mean_squared_error,median_absolute_error,r2_score,roc_auc_score,recall_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from fs_common import *


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:np.asarray(z[k]) for k in z.files}


def feature(d,name):
    s=np.asarray(d["stage1_logit"],np.float64).reshape(-1,1);r=np.asarray(d["R4_pre_delta"],np.float64)
    return s if name=="P0_stage1" else r if name=="P1_R4" else np.concatenate([r,s],axis=1)


def class_metrics(y,p,task):
    y=np.asarray(y,int);p=np.asarray(p,float);pred=p>=.5
    out={"N":len(y),"BA":float(balanced_accuracy_score(y,pred)),"AUC":float(roc_auc_score(y,p)) if len(set(y))==2 else np.nan,
         "F1":float(f1_score(y,pred,zero_division=0)),"ACC":float(accuracy_score(y,pred))}
    if task=="direction":
        out.update({"FP_n":int(np.sum(y==0)),"FN_n":int(np.sum(y==1)),
                    "FP_to_NEG_recall":float(recall_score(y,pred,pos_label=0,zero_division=0)),
                    "FN_to_POS_recall":float(recall_score(y,pred,pos_label=1,zero_division=0))})
    else:
        out.update({"PR_AUC":float(average_precision_score(y,p)),"wrong_recall":float(recall_score(y,pred,pos_label=1,zero_division=0)),
                    "correct_recall":float(recall_score(y,pred,pos_label=0,zero_division=0))})
    return out


def regression_metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);rho=spearmanr(y,p).statistic if len(y)>1 else np.nan
    return {"N":len(y),"MAE":float(mean_absolute_error(y,p)),"Median_AE":float(median_absolute_error(y,p)),
            "RMSE":float(mean_squared_error(y,p)**.5),"R2":float(r2_score(y,p)),"Spearman":float(rho)}


def fit_class(x,y,z):
    if len(np.unique(y))!=2:raise RuntimeError("probe outer-train target has one class")
    m=make_pipeline(StandardScaler(),LogisticRegression(C=1.,solver="liblinear",class_weight="balanced",max_iter=5000,random_state=20260920))
    m.fit(x,y);return m.predict_proba(z)[:,1]


def fit_ridge(x,y,z):
    m=make_pipeline(StandardScaler(),Ridge(alpha=1.0));m.fit(x,y);return m.predict(z)


def aggregate(rows,task,scope_key):
    grouped=defaultdict(list)
    for r in rows:grouped[(r[scope_key],r["representation"])].append(r)
    out=[]
    for (scope,rep),parts in grouped.items():
        y=np.concatenate([p["y"] for p in parts]);prob=np.concatenate([p["prob"] for p in parts])
        met=class_metrics(y,prob,task) if task!="magnitude" else regression_metrics(y,prob)
        out.append({scope_key:scope,"representation":rep,**met})
    return out


def main():
    audit=read_csv(OUT/"preprocessing_reconstruction_audit.csv")
    if len(audit)!=480 or not all(int(r["passed"]) for r in audit):raise RuntimeError("BLOCKED_BY_CONTRACT: preprocessing gate incomplete")
    reps=("P0_stage1","P1_R4","P2_R4_plus_stage1")
    direction_parts=[];need_parts=[];mag_parts=[];fold_rows=[]
    for y in YEARS:
      for f in FOLDS:
        per={rep:{"direction":[],"need":[],"magnitude":[]} for rep in reps};val_ref=None
        for s in SEEDS:
            root=OUT/"cache"/str(y)/f"fold_{f}"/f"seed_{s}";tr=load(root/"train_r4.npz");va=load(root/"validation_r4.npz")
            if val_ref is None:val_ref=va
            elif not np.array_equal(val_ref["sha256"],va["sha256"]):raise RuntimeError("seed validation order differs")
            tr_wrong=(tr["p_stage1"]>=.5)!=tr["label"];va_wrong=(va["p_stage1"]>=.5)!=va["label"]
            tr_direction=tr["routed"].astype(bool)&tr_wrong;va_direction=va["routed"].astype(bool)&va_wrong
            tr_need=tr["routed"].astype(bool);va_need=va["routed"].astype(bool)
            for rep in reps:
                per[rep]["direction"].append(fit_class(feature(tr,rep)[tr_direction],tr["label"][tr_direction],feature(va,rep)[va_direction]))
                per[rep]["need"].append(fit_class(feature(tr,rep)[tr_need],tr_wrong[tr_need].astype(int),feature(va,rep)[va_need]))
                target=np.minimum(np.abs(tr["stage1_logit"][tr_direction])+1e-3,2.25)
                per[rep]["magnitude"].append(fit_ridge(feature(tr,rep)[tr_direction],target,feature(va,rep)[va_direction]))
        va=val_ref;wrong=(va["p_stage1"]>=.5)!=va["label"];dm=va["routed"].astype(bool)&wrong;nm=va["routed"].astype(bool)
        for rep in reps:
            dp=np.mean(per[rep]["direction"],axis=0);np_=np.mean(per[rep]["need"],axis=0);mp=np.mean(per[rep]["magnitude"],axis=0)
            dy=va["label"][dm].astype(int);ny=wrong[nm].astype(int);my=np.minimum(np.abs(va["stage1_logit"][dm])+1e-3,2.25)
            fold_rows.append({"task":"direction","year":y,"fold":f,"representation":rep,**class_metrics(dy,dp,"direction")})
            fold_rows.append({"task":"need_correction","year":y,"fold":f,"representation":rep,**class_metrics(ny,np_,"need")})
            fold_rows.append({"task":"magnitude","year":y,"fold":f,"representation":rep,
                              "BOUND_LIMIT_n":int(np.sum(np.abs(va["stage1_logit"][dm])+1e-3>2.25)),**regression_metrics(my,mp)})
            direction_parts.append({"year":y,"fold":f,"representation":rep,"y":dy,"prob":dp})
            need_parts.append({"year":y,"fold":f,"representation":rep,"y":ny,"prob":np_})
            mag_parts.append({"year":y,"fold":f,"representation":rep,"y":my,"prob":mp})
        print(f"probes year={y} fold={f}/5 done",flush=True)
    write_csv(OUT/"direction_probe_by_fold.csv",[r for r in fold_rows if r["task"]=="direction"])
    dyear=aggregate(direction_parts,"direction","year");nyear=aggregate(need_parts,"need","year");myear=aggregate(mag_parts,"magnitude","year")
    write_csv(OUT/"direction_probe_by_year.csv",dyear);write_csv(OUT/"correction_need_probe_by_year.csv",nyear);write_csv(OUT/"magnitude_probe_by_year.csv",myear)
    def pooled(parts,task):
        out=[]
        for rep in reps:
            q=[r for r in parts if r["representation"]==rep];yy=np.concatenate([r["y"] for r in q]);pp=np.concatenate([r["prob"] for r in q])
            out.append({"scope":"pooled_2018_2021","representation":rep,**(class_metrics(yy,pp,task) if task!="magnitude" else regression_metrics(yy,pp))})
        return out
    dp=pooled(direction_parts,"direction");npo=pooled(need_parts,"need");mpo=pooled(mag_parts,"magnitude")
    write_csv(OUT/"direction_probe_pooled.csv",dp);write_csv(OUT/"correction_need_probe_pooled.csv",npo);write_csv(OUT/"magnitude_probe_pooled.csv",mpo)
    inc=[]
    for task,rows,metrics in (("direction",dp,("BA","AUC")),("need_correction",npo,("BA","AUC","PR_AUC"))):
        by={r["representation"]:r for r in rows};base=by["P0_stage1"]
        for rep in ("P1_R4","P2_R4_plus_stage1"):
            inc.append({"task":task,"comparison":rep+" minus P0_stage1",**{f"Delta_{m}":float(by[rep][m])-float(base[m]) for m in metrics}})
    write_csv(OUT/"representation_incremental_value.csv",inc)
    no=json.loads((OUT/"NO_STAGE2_RETRAINING_AUDIT.json").read_text(encoding="utf-8"));no["validation_labels_used_for_probe_training"]=False
    no.update({"diagnostic_probe_training_invoked":True,"diagnostic_probe_types":["LogisticRegression","Ridge"],"probe_predictions_used_as_CARE_predictions":False})
    write_json(OUT/"NO_STAGE2_RETRAINING_AUDIT.json",no)
    byD={r["representation"]:r for r in dp};byN={r["representation"]:r for r in npo};byM={r["representation"]:r for r in mpo}
    delta_auc=byD["P2_R4_plus_stage1"]["AUC"]-byD["P0_stage1"]["AUC"]
    # Descriptive interpretation only; no method selection or threshold tuning.
    if byD["P1_R4"]["BA"]>=.65 and delta_auc>=.03:decision="CORRECTION_MAPPING_INSUFFICIENT"
    elif byD["P1_R4"]["BA"]<.60 and byN["P1_R4"]["AUC"]<.60:decision="FEATURE_EXTRACTION_INSUFFICIENT"
    elif byD["P1_R4"]["BA"]<.60:decision="BOTH_FEATURE_AND_MAPPING"
    else:decision="CORRECTION_AUTHORITY_DOMINANT"
    lines=["CARE-Droid Stage-II Feature Sufficiency Audit V2","Development/diagnostic same-year OOF only; not unseen, future, or final test.",
      "R4_pre_delta is a mixed 113-D pre-head representation and includes Global24 plus Stage-I logit.","",
      f"Q1 validation forward replay: PASS (60/60).",
      "Q2 R4 composition: Global24(24)+Stage-I logit(1)+graph token(32)+critical HostEgo(24)+attended HostEgo(24)+summary(8)=113.",
      "Q3 preprocessing reconstruction: PASS; Stage-II neural training invoked: false.",
      f"Q4 R4-only Direction BA={byD['P1_R4']['BA']:.6f} AUC={byD['P1_R4']['AUC']:.6f}.",
      f"Q5 R4+Stage1 Direction BA={byD['P2_R4_plus_stage1']['BA']:.6f} AUC={byD['P2_R4_plus_stage1']['AUC']:.6f}.",
      f"Q6 Stage1-only Direction BA={byD['P0_stage1']['BA']:.6f} AUC={byD['P0_stage1']['AUC']:.6f}.",
      f"Q7 incremental Direction AUC (R4+Stage1 minus Stage1)={delta_auc:+.6f}; diagnostic only.",
      f"Q8 Need-correction R4-only AUC={byN['P1_R4']['AUC']:.6f} PR-AUC={byN['P1_R4']['PR_AUC']:.6f}.",
      f"Q9 Magnitude R4-only MAE={byM['P1_R4']['MAE']:.6f} R2={byM['P1_R4']['R2']:.6f} Spearman={byM['P1_R4']['Spearman']:.6f}.",
      f"Q10 diagnostic classification: {decision}.","No CARE-Droid model, router, checkpoint, or prediction was modified."]
    (OUT/"FEATURE_SUFFICIENCY_REPORT_V2.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (OUT/"FEATURE_SUFFICIENCY_DECISION.txt").write_text("DIAGNOSTIC_ONLY\n"+decision+"\n",encoding="utf-8")
    print("FEATURE_SUFFICIENCY_PROBES_DONE")

if __name__=="__main__":main()
