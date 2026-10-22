#!/usr/bin/env python3
from __future__ import annotations

import csv, json, math, sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"care-p31rf-c43-apiidentity-probe-source-v1"))
from api_identity_core import FoldAPIProjector, build_api_bags, norm_sha, read_csv, read_endpoint_cache, write_csv
from dependency_core import TrainOnlyProjector, verified_matrix, role_cooccurrence_matrix, stratified_shuffle

ROOT=Path("artifacts/care-p31rf-c43-dependency-sufficiency-source-v1")
REF=Path("artifacts/care-p31-stage1-rf-selection-source-v1/preflight/C_RF_RFROUTE/source_predictions.csv")
PRED=Path("artifacts/care-p31-stage1-rf-selection-source-v1/arms/C_RF_RFROUTE/source_predictions.csv")
ENDPOINT=Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz")
P71=Path("artifacts/p71/verified_dependency_features_2017_v3.csv")
ROLE_NAMES=("command_execution","dynamic_code_loading","native_library_loading","reflective_invocation","package_installation","package_uninstallation","device_administration","accessibility_control","sms_send_receive","call_control_or_recording","account_or_credential_access","contacts_or_calendar_access","device_identifier_access","location_access","microphone_access","camera_access","clipboard_access","network_transmission_or_upload","sensitive_file_or_content_write","boot_persistence","task_or_alarm_scheduling","background_service_control","overlay_or_ui_deception","icon_or_component_hiding","process_or_wakelock_control","cryptography","encoding_or_compression","url_or_http_setup","file_read","database_access","other")
METHODS=("D0_ROLE","D1_ROLE_API","D2_ROLE_COOCCURRENCE","D3_ROLE_VERIFIED_DEP","D4_COUNT_ONLY","D5_ROLE_SHUFFLED_DEP")

def logit(p):
    p=np.clip(np.asarray(p,float),1e-7,1-1e-7);return np.log(p/(1-p))

def fit_probe(x,y,train,val,seed):
    model=Pipeline([("scale",StandardScaler()),("model",LogisticRegression(C=1.0,solver="liblinear",class_weight="balanced",max_iter=5000,random_state=seed))])
    model.fit(x[train],y[train]);return np.clip(model.predict_proba(x[val])[:,1],1e-7,1-1e-7)

def auc(y,p):return float(roc_auc_score(y,p)) if len(np.unique(y))==2 else float("nan")
def metrics(y,p):return {"n":len(y),"BA":float(balanced_accuracy_score(y,p>=.5)),"AUC":auc(y,p),"F1":float(f1_score(y,p>=.5,zero_division=0))}

def direction(name,y,base,route,score,fold):
    err=route&((base>=.5)!=y); truth=np.where(y==1,1.,-1.); aligned=logit(score)*truth
    correct=aligned>1e-12; amb=np.abs(aligned)<=1e-12; fn=err&(y==1); fp=err&(y==0)
    byfold=[float(np.mean(correct[err&(fold==f)])) for f in range(1,6)]
    return {"Representation":name,"RoutedErrors":int(err.sum()),"FN":int(fn.sum()),"FP":int(fp.sum()),
      "OverallDirectionAcc":float(np.mean(correct[err])),"FNDirectionAcc":float(np.mean(correct[fn])),"FPDirectionAcc":float(np.mean(correct[fp])),
      "WrongDirection":int(np.sum(err&~correct&~amb)),"Ambiguous":int(np.sum(err&amb)),"CorrectDirection":int(np.sum(err&correct)),
      "MeanAligned":float(np.mean(aligned[err])),"MedianAligned":float(np.median(aligned[err])),
      "FNMeanAligned":float(np.mean(aligned[fn])),"FPMeanAligned":float(np.mean(aligned[fp])),
      "FoldMean":float(np.mean(byfold)),"FoldStd":float(np.std(byfold)),"DirectionAUC":auc(y[err],score[err])}

def paired(y,base,route,a,b,name):
    s=route&((base>=.5)!=y); ac=(a[s]>=.5)==y[s];bc=(b[s]>=.5)==y[s]
    n01=int(np.sum(ac&~bc));n10=int(np.sum(~ac&bc));n=n01+n10
    return {"comparison":name,"candidate_only_correct":n01,"control_only_correct":n10,"paired_net":n01-n10,"exact_p":float(binomtest(min(n01,n10),n,.5).pvalue) if n else 1.}

def find_final(rows):
    preferred="P31RF__RFRoute__P0Zero6C43RQ__Bounded__prob"
    if preferred in rows[0]:return preferred
    found=[k for k in rows[0] if k.endswith("__prob") and "P31RF" in k and "Bounded" in k]
    if len(found)!=1:raise RuntimeError("BLOCKED_BY_CONTRACT: frozen RF-C43 column ambiguous")
    return found[0]

def main():
    ref=read_csv(REF); ids=[norm_sha(r["sha256"]) for r in ref]; pred_by={norm_sha(r["sha256"]):r for r in read_csv(PRED)}; pred=[pred_by[x] for x in ids]
    y=np.array([int(float(r["label"])) for r in ref]);fold=np.array([int(float(r["fold"])) for r in ref])
    base=np.array([float(r["Global24__prob"]) for r in ref]);route=np.array([int(float(r["A25__routed"]))==1 for r in ref]);base_pred=(base>=.5).astype(int)
    final=np.array([float(r[find_final(pred)]) for r in pred]);current_delta=logit(final)-logit(base);current_score=1/(1+np.exp(-current_delta))
    endpoint,load_s=read_endpoint_cache(ENDPOINT);bags,audits,failures,rebuild_s=build_api_bags(ref,endpoint)
    role=np.zeros((799,31),float)
    for i,bag in enumerate(bags):
        if bag:role[i]=np.mean([h["role"] for h in bag],axis=0)
    cooc,cooc_names=role_cooccurrence_matrix(bags)
    dep_rows=read_csv(P71);dep,dep_names,dep_status=verified_matrix(ids,dep_rows,ROLE_NAMES)
    oof={m:np.full(799,np.nan) for m in METHODS};norms={m:np.full(799,np.nan) for m in METHODS};fold_audit=[]
    for f in range(1,6):
        tr=np.flatnonzero(fold!=f);va=np.flatnonzero(fold==f);seed=29000+f
        api=FoldAPIProjector(seed=seed).fit([bags[i] for i in tr]);api_tr=api.transform([bags[i] for i in tr])["api"];api_va=api.transform([bags[i] for i in va])["api"]
        cp=TrainOnlyProjector(seed=seed).fit(cooc[tr],cooc_names);dp=TrainOnlyProjector(seed=seed+100).fit(dep[tr],dep_names)
        co_tr,co_va=cp.transform(cooc[tr]),cp.transform(cooc[va]);de_tr,de_va=dp.transform(dep[tr]),dp.transform(dep[va])
        sh_tr=stratified_shuffle(de_tr,base_pred[tr],seed+200);sh_va=stratified_shuffle(de_va,base_pred[va],seed+300)
        count=np.column_stack(([a["host_count"] for a in audits],[a["api_occurrence_count"] for a in audits],[a["unique_api_count"] for a in audits],np.count_nonzero(dep,axis=1)))
        xtr={"D0_ROLE":np.c_[role[tr],np.zeros((len(tr),31))],"D1_ROLE_API":np.c_[role[tr],api_tr],"D2_ROLE_COOCCURRENCE":np.c_[role[tr],co_tr],"D3_ROLE_VERIFIED_DEP":np.c_[role[tr],de_tr],"D4_COUNT_ONLY":count[tr],"D5_ROLE_SHUFFLED_DEP":np.c_[role[tr],sh_tr]}
        xva={"D0_ROLE":np.c_[role[va],np.zeros((len(va),31))],"D1_ROLE_API":np.c_[role[va],api_va],"D2_ROLE_COOCCURRENCE":np.c_[role[va],co_va],"D3_ROLE_VERIFIED_DEP":np.c_[role[va],de_va],"D4_COUNT_ONLY":count[va],"D5_ROLE_SHUFFLED_DEP":np.c_[role[va],sh_va]}
        for m in METHODS:
            x=np.r_[xtr[m],xva[m]];oof[m][va]=fit_probe(x,np.r_[y[tr],y[va]],np.arange(len(tr)),np.arange(len(tr),len(tr)+len(va)),seed);norms[m][va]=np.linalg.norm(xva[m],axis=1)
        fold_audit.append({"fold":f,"train_rows":len(tr),"validation_rows":len(va),"api_vocab":len(api.vocabulary),"cooccurrence_active_train_columns":len(cp.active),"cooccurrence_components":cp.components,"dependency_active_train_columns":len(dp.active),"dependency_components":dp.components,"dependency_schema_hash":dp.schema_hash})
        print(f"dependency sufficiency fold {f}/5 done dep_active={len(dp.active)} cooc_active={len(cp.active)}",flush=True)
    if any(not np.isfinite(oof[m]).all() for m in METHODS):raise RuntimeError("incomplete OOF dependency probe")
    quality=[direction("E_CURRENT_C43_DELTA",y,base,route,current_score,fold)]+[direction(m,y,base,route,oof[m],fold) for m in METHODS]
    err=route&((base>=.5)!=y);perfold=[]
    for f in range(1,6):
        s=err&(fold==f);row={"Fold":f,"RoutedErrors":int(s.sum()),"FN":int(np.sum(s&(y==1))),"FP":int(np.sum(s&(y==0)))}
        for m in METHODS:row[m+"_acc"]=float(np.mean((oof[m][s]>=.5)==y[s]))
        perfold.append(row)
    subgroup=[]
    for comparison,subset in (("FP_vs_TP",route&(base>=.5)),("FN_vs_TN",route&(base<.5))):
        for m in METHODS:subgroup.append({"Comparison":comparison,"Representation":m,**metrics(y[subset],oof[m][subset])})
    comparisons=[paired(y,base,route,oof["D3_ROLE_VERIFIED_DEP"],oof[x],"D3_minus_"+x) for x in ("D0_ROLE","D1_ROLE_API","D2_ROLE_COOCCURRENCE","D5_ROLE_SHUFFLED_DEP")]
    shortcut=[]
    variables={"HostCount":np.array([a["host_count"] for a in audits]),"APICount":np.array([a["api_occurrence_count"] for a in audits]),"UniqueAPICount":np.array([a["unique_api_count"] for a in audits]),"GraphNodeCount":np.array([a["graph_node_count"] for a in audits]),"DependencyNonzeroCount":np.count_nonzero(dep,axis=1)}
    for m in METHODS:
        for name,value in variables.items():
            shortcut.append({"Representation":m,"AuditVariable":name,"corr_feature_norm":float(spearmanr(norms[m],value).statistic),"corr_score":float(spearmanr(oof[m],value).statistic)})
    write_csv(ROOT/"direction_quality.csv",quality);write_csv(ROOT/"per_fold_direction.csv",perfold);write_csv(ROOT/"failure_taxonomy.csv",[{"Representation":r["Representation"],"WrongDirection":r["WrongDirection"],"Ambiguous":r["Ambiguous"],"CorrectDirection":r["CorrectDirection"]} for r in quality]);write_csv(ROOT/"fp_tp_fn_tn_metrics.csv",subgroup);write_csv(ROOT/"paired_direction.csv",comparisons);write_csv(ROOT/"shortcut_audit.csv",shortcut);write_csv(ROOT/"fold_audit.csv",fold_audit);write_csv(ROOT/"api_reconstruction_failures.csv",failures)
    prediction=[]
    for i,sha in enumerate(ids):
        row={"sha256":sha,"fold":int(fold[i]),"label":int(y[i]),"stage1_prob":float(base[i]),"stage1_pred":int(base_pred[i]),"route":int(route[i]),"dependency_analysis_status":dep_status[i],"dependency_nonzero":int(np.count_nonzero(dep[i])),**audits[i]}
        for m in METHODS:row[m+"__prob"]=float(oof[m][i]);row[m+"__feature_norm"]=float(norms[m][i])
        prediction.append(row)
    write_csv(ROOT/"source_oof_predictions.csv",prediction)
    q={r["Representation"]:r for r in quality};sub={(r["Comparison"],r["Representation"]):r for r in subgroup};d0=q["D0_ROLE"];d3=q["D3_ROLE_VERIFIED_DEP"]
    fold_gain=[r["D3_ROLE_VERIFIED_DEP_acc"]-r["D0_ROLE_acc"] for r in perfold]
    advance=(d3["FPDirectionAcc"]-d0["FPDirectionAcc"]>=.10 and d3["FNDirectionAcc"]>=d0["FNDirectionAcc"]-.05 and d3["OverallDirectionAcc"]>d0["OverallDirectionAcc"] and sub[("FP_vs_TP","D3_ROLE_VERIFIED_DEP")]["AUC"]>.55 and d3["OverallDirectionAcc"]>q["D2_ROLE_COOCCURRENCE"]["OverallDirectionAcc"] and d3["OverallDirectionAcc"]>q["D5_ROLE_SHUFFLED_DEP"]["OverallDirectionAcc"] and sum(v>=0 for v in fold_gain)>=3)
    decision={"advance_to_host_aligned_exact_api_dependency_extraction":bool(advance),"advance_directly_to_c43":False,"observed":{"FP_direction_gain":d3["FPDirectionAcc"]-d0["FPDirectionAcc"],"FN_direction_change":d3["FNDirectionAcc"]-d0["FNDirectionAcc"],"overall_direction_gain":d3["OverallDirectionAcc"]-d0["OverallDirectionAcc"],"FP_vs_TP_AUC":sub[("FP_vs_TP","D3_ROLE_VERIFIED_DEP")]["AUC"],"nonnegative_folds":sum(v>=0 for v in fold_gain)},"next_step":"HOST_ALIGNED_EXACT_API_DEPENDENCY_EXTRACTION" if advance else "STOP_VERIFIED_ROLE_DEPENDENCY_BRANCH","evidence_boundary":"P71 is role-pair dependency evidence at APK level; it does not preserve exact source API identity or C43 Host ownership."}
    (ROOT/"advance_decision.json").write_text(json.dumps(decision,indent=2)+"\n",encoding="utf-8")
    coverage={"source_rows":799,"routed":int(route.sum()),"routed_errors":int(err.sum()),"FN":int(np.sum(err&(y==1))),"FP":int(np.sum(err&(y==0))),"dependency_rows":len(dep_rows),"dependency_rows_with_nonzero_verified_feature":int(np.sum(np.count_nonzero(dep,axis=1)>0)),"predeclared_dependency_width":len(dep_names),"observed_nonzero_dependency_columns":int(np.sum(np.any(dep>0,axis=0))),"endpoint_cache_load_seconds":load_s,"host_reconstruction_seconds":rebuild_s,"future_rows_loaded":0}
    (ROOT/"coverage_cost.json").write_text(json.dumps(coverage,indent=2)+"\n",encoding="utf-8")
    report=["CARE-Droid P31-RF verified-dependency sufficiency probe source v1","","Strict 2017 five-fold OOF; fixed P31-RF and RF-anchored DI25=200/799.","No C43 training/change, no future row. Co-occurrence is an explicit negative control, not dependency.","P71 dependency evidence is conservative but role-pair/APK-level; exact source API and C43 Host ownership are not persisted.",""]
    for r in quality:report.append(f"{r['Representation']}: overall={r['OverallDirectionAcc']:.6f} FN={r['FNDirectionAcc']:.6f} FP={r['FPDirectionAcc']:.6f} mean_aligned={r['MeanAligned']:.6f} fold_std={r['FoldStd']:.6f}")
    report += ["",f"DEPENDENCY_SUFFICIENCY_ADVANCE={'PASS' if advance else 'STOP'}",f"Next step: {decision['next_step']}","A PASS authorizes only Host-aligned exact-API dependency extraction; it does not authorize C43 integration or future diagnostics."]
    (ROOT/"REPORT.txt").write_text("\n".join(report)+"\n",encoding="utf-8")
    print("P31RF_DEPENDENCY_SUFFICIENCY_PROBE_DONE")
    print("DEPENDENCY_SUFFICIENCY_ADVANCE="+("PASS" if advance else "STOP"))

if __name__=="__main__":main()

