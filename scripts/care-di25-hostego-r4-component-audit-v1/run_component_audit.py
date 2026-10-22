#!/usr/bin/env python3
"""Read-only component audit over already frozen R4 train/validation caches."""
from __future__ import annotations
import csv,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.metrics import (average_precision_score,balanced_accuracy_score,f1_score,mean_absolute_error,
                             mean_squared_error,median_absolute_error,r2_score,recall_score,roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent
V2=HERE.parent/"care-di25-hostego-feature-sufficiency-v2"
sys.path.insert(0,str(V2))
from fs_common import OUT as V2OUT, YEARS,FOLDS,SEEDS,norm,read_csv,write_csv,write_json

OUT=Path("artifacts/care-di25-hostego-dmd-bounded-v1/r4_component_audit")
REPS=("S0_stage1_logit","S1_global24_plus_logit","S2_stage2_tail88","S3_tail88_plus_logit","S4_full_R4")
BLOCKS={"B0_graph_token32":(25,57),"B1_critical_hostego24":(57,81),
        "B2_attended_hostego24":(81,105),"B3_attention_anomaly_summary8":(105,113)}
RNG_SEED=20260920

def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:np.asarray(z[k]) for k in z.files}

def features(d,name):
    r=np.asarray(d["R4_pre_delta"],float);logit=np.asarray(d["stage1_logit"],float).reshape(-1,1)
    if r.shape[1]!=113:raise RuntimeError("R4 dimension changed")
    if name=="S0_stage1_logit":return logit
    if name=="S1_global24_plus_logit":return r[:,:25]
    if name=="S2_stage2_tail88":return r[:,25:]
    if name=="S3_tail88_plus_logit":return np.concatenate([r[:,25:],logit],axis=1)
    if name=="S4_full_R4":return r
    if name in BLOCKS:
        a,b=BLOCKS[name];return r[:,a:b]
    raise KeyError(name)

def fit_class(x,y,z):
    if len(np.unique(y))!=2:raise RuntimeError("outer-train probe target has one class")
    m=make_pipeline(StandardScaler(),LogisticRegression(C=1.,solver="liblinear",class_weight="balanced",max_iter=5000,random_state=RNG_SEED))
    m.fit(x,y);return m.predict_proba(z)[:,1]

def fit_reg(x,y,z):
    m=make_pipeline(StandardScaler(),Ridge(alpha=1.0));m.fit(x,y);return m.predict(z)

def metrics(y,p,task):
    y=np.asarray(y,int);p=np.asarray(p,float);pred=p>=.5
    present=set(y.tolist());two_classes=present=={0,1}
    out={"N":len(y),"class_0_n":int(np.sum(y==0)),"class_1_n":int(np.sum(y==1)),
         "BA":float(balanced_accuracy_score(y,pred)) if two_classes else np.nan,
         "AUC":float(roc_auc_score(y,p)) if two_classes else np.nan,
         "F1":float(f1_score(y,pred,zero_division=0))}
    recall0=float(recall_score(y,pred,pos_label=0,zero_division=0)) if 0 in present else np.nan
    recall1=float(recall_score(y,pred,pos_label=1,zero_division=0)) if 1 in present else np.nan
    if task=="direction":out.update({"FP_to_NEG_recall":recall0,"FN_to_POS_recall":recall1})
    else:out.update({"PR_AUC":float(average_precision_score(y,p)) if two_classes else np.nan,
                     "wrong_recall":recall1,"correct_recall":recall0})
    return out

def reg_metrics(y,p):
    rho=spearmanr(y,p).statistic
    return {"N":len(y),"MAE":float(mean_absolute_error(y,p)),"Median_AE":float(median_absolute_error(y,p)),
            "RMSE":float(mean_squared_error(y,p)**.5),"R2":float(r2_score(y,p)),"Spearman":float(rho)}

def bootstrap_delta(y,pa,pb,task,iterations=10000):
    y=np.asarray(y);pa=np.asarray(pa);pb=np.asarray(pb);rng=np.random.default_rng(RNG_SEED);values={"BA":[],"AUC":[]}
    for _ in range(iterations):
        idx=rng.integers(0,len(y),len(y));yy=y[idx]
        if len(np.unique(yy))<2:continue
        for key in values:
            fun=(lambda q:balanced_accuracy_score(yy,q>=.5)) if key=="BA" else (lambda q:roc_auc_score(yy,q))
            values[key].append(fun(pb[idx])-fun(pa[idx]))
    row={"bootstrap_iterations_requested":iterations,"bootstrap_seed":RNG_SEED}
    for key,v in values.items():
        a=np.asarray(v,float);row.update({f"Delta_{key}_mean":float(np.mean(a)),f"Delta_{key}_ci_low":float(np.percentile(a,2.5)),f"Delta_{key}_ci_high":float(np.percentile(a,97.5)),f"valid_{key}_iterations":len(a)})
    return row

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    schema=json.loads((V2OUT/"pre_delta_feature_schema.json").read_text(encoding="utf-8"))
    expected=[("global24_features",24),("stage1_base_logit",1),("role_graph_token",32),("critical_hostego_embedding",24),("attention_weighted_hostego_embedding",24),("scalar_summary",8)]
    actual=[(x["name"],int(x["dimension"])) for x in schema["components"]]
    if schema["status"].startswith("NOT") or actual!=expected:raise RuntimeError("BLOCKED_BY_SCHEMA: pre-delta layout differs")
    cache_missing=[];max_logit_diff=0.;cache_n=0
    for y in YEARS:
      for f in FOLDS:
       for s in SEEDS:
        for split in ("train","validation"):
          p=V2OUT/"cache"/str(y)/f"fold_{f}"/f"seed_{s}"/f"{split}_r4.npz"
          if not p.is_file():cache_missing.append(str(p));continue
          d=load(p);cache_n+=1;max_logit_diff=max(max_logit_diff,float(np.max(np.abs(d["R4_pre_delta"][:,24]-d["stage1_logit"]))))
    integrity={"expected_cache_files":120,"found_cache_files":cache_n,"missing":cache_missing,"R4_column24_equals_stage1_logit_max_abs_diff":max_logit_diff,
               "passed":cache_n==120 and not cache_missing and max_logit_diff<=1e-6}
    write_json(OUT/"r4_component_integrity.json",integrity)
    if not integrity["passed"]:raise RuntimeError("BLOCKED_BY_CACHE_INTEGRITY")

    all_reps=REPS+tuple(BLOCKS);pred_rows=[];fold_metrics=[];magnitude_rows=[];mag_pred_rows=[]
    for y in YEARS:
      for f in FOLDS:
        seed_probs={task:{rep:[] for rep in all_reps} for task in ("direction","need")};mag_probs=defaultdict(list);va_ref=None
        for s in SEEDS:
          root=V2OUT/"cache"/str(y)/f"fold_{f}"/f"seed_{s}";tr=load(root/"train_r4.npz");va=load(root/"validation_r4.npz")
          if va_ref is None:va_ref=va
          elif not np.array_equal(va_ref["sha256"],va["sha256"]):raise RuntimeError("seed validation order differs")
          tw=(tr["p_stage1"]>=.5)!=tr["label"];vw=(va["p_stage1"]>=.5)!=va["label"]
          td=tr["routed"].astype(bool)&tw;vd=va["routed"].astype(bool)&vw;tn=tr["routed"].astype(bool);vn=va["routed"].astype(bool)
          for rep in all_reps:
            seed_probs["direction"][rep].append(fit_class(features(tr,rep)[td],tr["label"][td],features(va,rep)[vd]))
            seed_probs["need"][rep].append(fit_class(features(tr,rep)[tn],tw[tn].astype(int),features(va,rep)[vn]))
          target=np.minimum(np.abs(tr["stage1_logit"][td])+1e-3,2.25);val_target=np.minimum(np.abs(va["stage1_logit"][vd])+1e-3,2.25)
          mag_defs={"M0_signed_logit":tr["stage1_logit"].reshape(-1,1),"M1_abs_logit":np.abs(tr["stage1_logit"]).reshape(-1,1),
                    "M2_tail88":features(tr,"S2_stage2_tail88"),"M3_tail88_plus_abslogit":np.concatenate([features(tr,"S2_stage2_tail88"),np.abs(tr["stage1_logit"]).reshape(-1,1)],axis=1)}
          val_defs={"M0_signed_logit":va["stage1_logit"].reshape(-1,1),"M1_abs_logit":np.abs(va["stage1_logit"]).reshape(-1,1),
                    "M2_tail88":features(va,"S2_stage2_tail88"),"M3_tail88_plus_abslogit":np.concatenate([features(va,"S2_stage2_tail88"),np.abs(va["stage1_logit"]).reshape(-1,1)],axis=1)}
          for name in mag_defs:mag_probs[name].append(fit_reg(mag_defs[name][td],target,val_defs[name][vd]))
        va=va_ref;wrong=(va["p_stage1"]>=.5)!=va["label"];dm=va["routed"].astype(bool)&wrong;nm=va["routed"].astype(bool)
        for task,mask,target in (("direction",dm,va["label"][dm]),("need",nm,wrong[nm].astype(int))):
          for rep in all_reps:
            p=np.mean(seed_probs[task][rep],axis=0);fold_metrics.append({"task":task,"year":y,"fold":f,"representation":rep,**metrics(target,p,task)})
            for i,sha in enumerate(va["sha256"][mask]):pred_rows.append({"task":task,"year":y,"fold":f,"sha256":norm(sha),"label":int(target[i]),"representation":rep,"probability":float(p[i])})
        val_target=np.minimum(np.abs(va["stage1_logit"][dm])+1e-3,2.25)
        for name,parts in mag_probs.items():
          p=np.mean(parts,axis=0);m=reg_metrics(val_target,p);magnitude_rows.append({"year":y,"fold":f,"representation":name,**m})
          for i,sha in enumerate(va["sha256"][dm]):
            mag_pred_rows.append({"year":y,"fold":f,"sha256":norm(sha),"representation":name,
                                  "target":float(val_target[i]),"prediction":float(p[i])})
        print(f"R4 component audit {y} fold={f}/5 done",flush=True)
    write_csv(OUT/"component_probe_by_fold.csv",fold_metrics);write_csv(OUT/"component_probe_oof_predictions.csv",pred_rows);write_csv(OUT/"magnitude_sanity_by_fold.csv",magnitude_rows)

    pooled=[];by_pred={}
    for task in ("direction","need"):
      for rep in all_reps:
        q=[r for r in pred_rows if r["task"]==task and r["representation"]==rep];yy=np.asarray([int(r["label"]) for r in q]);pp=np.asarray([float(r["probability"]) for r in q])
        pooled.append({"task":task,"representation":rep,**metrics(yy,pp,task)});by_pred[(task,rep)]=(yy,pp)
    write_csv(OUT/"component_probe_pooled.csv",pooled)
    year_rows=[]
    for task in ("direction","need"):
      for year in YEARS:
       for rep in all_reps:
        q=[r for r in pred_rows if r["task"]==task and r["year"]==year and r["representation"]==rep]
        yy=np.asarray([int(r["label"]) for r in q]);pp=np.asarray([float(r["probability"]) for r in q])
        year_rows.append({"task":task,"year":year,"representation":rep,**metrics(yy,pp,task)})
    write_csv(OUT/"component_probe_by_year.csv",year_rows)
    write_csv(OUT/"magnitude_sanity_oof_predictions.csv",mag_pred_rows)
    mag_pool=[]
    for name in ("M0_signed_logit","M1_abs_logit","M2_tail88","M3_tail88_plus_abslogit"):
      q=[r for r in mag_pred_rows if r["representation"]==name]
      yy=np.asarray([float(r["target"]) for r in q]);pp=np.asarray([float(r["prediction"]) for r in q])
      mag_pool.append({"representation":name,"aggregation":"exact pooled OOF predictions",**reg_metrics(yy,pp)})
    write_csv(OUT/"magnitude_target_sanity.csv",mag_pool)

    comparisons=[("direction","S0_stage1_logit","S2_stage2_tail88"),("direction","S0_stage1_logit","S3_tail88_plus_logit"),
                 ("direction","S1_global24_plus_logit","S4_full_R4"),("need","S0_stage1_logit","S2_stage2_tail88"),
                 ("need","S0_stage1_logit","S3_tail88_plus_logit"),("need","S1_global24_plus_logit","S4_full_R4")]
    boot=[]
    for task,a,b in comparisons:
        ya,pa=by_pred[(task,a)];yb,pb=by_pred[(task,b)]
        if not np.array_equal(ya,yb):raise RuntimeError("paired target order differs")
        boot.append({"task":task,"baseline":a,"candidate":b,**bootstrap_delta(ya,pa,pb,task)})
    write_csv(OUT/"paired_incremental_bootstrap.csv",boot)
    po={(r["task"],r["representation"]):r for r in pooled};mg={r["representation"]:r for r in mag_pool}
    d0,d2,d3,d1,d4=[po[("direction",x)] for x in ("S0_stage1_logit","S2_stage2_tail88","S3_tail88_plus_logit","S1_global24_plus_logit","S4_full_R4")]
    n0,n2,n3,n1,n4=[po[("need",x)] for x in ("S0_stage1_logit","S2_stage2_tail88","S3_tail88_plus_logit","S1_global24_plus_logit","S4_full_R4")]
    lines=["CARE-Droid R4 component sufficiency audit","Same-year development/diagnostic OOF only; no Stage-II replay, preprocessing, or neural training.","",
      f"Direction S0 Stage1: BA={d0['BA']:.6f} AUC={d0['AUC']:.6f}",f"Direction S2 structural tail88: BA={d2['BA']:.6f} AUC={d2['AUC']:.6f}",
      f"Direction S3 tail88+logit: BA={d3['BA']:.6f} AUC={d3['AUC']:.6f}",f"Direction S1 Global24+logit: BA={d1['BA']:.6f} AUC={d1['AUC']:.6f}",
      f"Direction S4 full R4: BA={d4['BA']:.6f} AUC={d4['AUC']:.6f}",
      f"Need S0 Stage1: AUC={n0['AUC']:.6f} PR-AUC={n0['PR_AUC']:.6f}",f"Need S2 structural tail88: AUC={n2['AUC']:.6f} PR-AUC={n2['PR_AUC']:.6f}",
      f"Need S3 tail88+logit: AUC={n3['AUC']:.6f} PR-AUC={n3['PR_AUC']:.6f}",f"Need S1 Global24+logit: AUC={n1['AUC']:.6f} PR-AUC={n1['PR_AUC']:.6f}",
      f"Need S4 full R4: AUC={n4['AUC']:.6f} PR-AUC={n4['PR_AUC']:.6f}",
      f"Magnitude signed-logit Ridge: MAE={mg['M0_signed_logit']['MAE']:.6f} R2={mg['M0_signed_logit']['R2']:.6f}",
      f"Magnitude abs-logit sanity baseline: MAE={mg['M1_abs_logit']['MAE']:.6f} R2={mg['M1_abs_logit']['R2']:.6f}",
      "Interpretation: direction labels on Stage-I errors are algebraically tied to the Stage-I decision sign. Stage1 direction performance is therefore a task-definition baseline, not local-evidence attribution.",
      "Magnitude target is a deterministic function of abs(Stage-I logit); abs-logit is the appropriate sanity baseline.",
      "See paired_incremental_bootstrap.csv before claiming structural incremental value. No CARE-Droid prediction was produced or changed."]
    (OUT/"R4_COMPONENT_AUDIT_REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    write_json(OUT/"NO_MODEL_MODIFICATION_AUDIT.json",{"stage2_forward_invoked":False,"stage2_training_invoked":False,"optimizer_step_count":0,
      "preprocessing_reconstructed":False,"existing_R4_cache_read_only":True,"diagnostic_probes_only":True,"CARE_predictions_modified":False,
      "validation_labels_used_for_probe_training":False,"year_used_as_feature":False})
    print("R4_COMPONENT_AUDIT_DONE")

if __name__=="__main__":main()
