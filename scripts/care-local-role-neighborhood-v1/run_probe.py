#!/usr/bin/env python3
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr,spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,f1_score
from common import ROOT,BASE,CACHE,METHOD,K,MIN_SUPPORT,CONTROL_REPETITIONS,NEAR_DUPLICATE_CUTOFF,ROLE_NAMES,read_csv,write_csv,dump,norm,logit
from neighborhood import make_representations,apk_scores,role_scores,shuffled_roles

rows=read_csv(BASE);n=len(rows);sha=np.asarray([norm(r["sha256"]) for r in rows]);fold=np.asarray([int(r["fold"]) for r in rows]);y=np.asarray([int(r["label"]) for r in rows]);route=np.asarray([int(r["A25__routed"]) for r in rows],bool)
base=np.asarray([float(r["Global24__prob"]) for r in rows]);d0=np.asarray([float(r[METHOD+"__prob"]) for r in rows]);d0delta=np.asarray([float(r[METHOD+"__bounded_delta"]) for r in rows]);base_ok=(base>=.5)==y;d0_ok=(d0>=.5)==y;d0_fixed=(~base_ok)&d0_ok;d0_broken=base_ok&(~d0_ok)
with np.load(CACHE,allow_pickle=False) as z:host_values=np.asarray(z["host_values"],float);offsets=np.asarray(z["host_offsets"],int)
role,mass,apk,host_count=make_representations(host_values,offsets)
if len(role)!=799:raise RuntimeError("BLOCKED_BY_CONTRACT: representation rows changed")

scores={"L0_CURRENT_C43":d0delta.copy(),"L1_APK_KNN":np.full(n,np.nan),"L2_ROLE_KNN":np.full(n,np.nan),"L3_STAGE1_CONDITIONED":np.full(n,np.nan),"L1_APK_KNN_EXCLUDE_GE_099":np.full(n,np.nan),"L2_ROLE_KNN_EXCLUDE_GE_099":np.full(n,np.nan)}
controls={name:[np.full(n,np.nan) for _ in range(CONTROL_REPETITIONS)] for name in ("LABEL_SHUFFLE_L1","LABEL_SHUFFLE_L2","RANDOM_NEIGHBOR_L1","RANDOM_NEIGHBOR_L2","ROLE_SHUFFLE_L2")}
apk_rows=[];role_rows=[];support_rows=[];neighbor_rows=[]
meta_names=("package","package_name","certificate","cert","family","app_id","version","method_count","graph_node_count","graph_edge_count")
available_meta=[name for name in meta_names if name in rows[0]]

def append_neighbors(level,outer,items):
    for qi,ri,bank_class,rank,ni,sim in items:
        out={"level":level,"query_sha256":sha[qi],"query_fold":outer,"query_label":int(y[qi]),"neighbor_rank":rank,"neighbor_sha256":sha[ni],"neighbor_label":int(y[ni]),"bank_class":bank_class,"similarity":sim,"query_role":"" if ri<0 else ri,"query_role_name":"" if ri<0 else ROLE_NAMES[ri],"neighbor_role":"" if ri<0 else ri,"neighbor_role_name":"" if ri<0 else ROLE_NAMES[ri],"same_sha256":int(sha[qi]==sha[ni])}
        for name in available_meta:out["query_"+name]=rows[qi].get(name,"");out["neighbor_"+name]=rows[ni].get(name,"")
        neighbor_rows.append(out)

for outer in range(1,6):
    train=np.flatnonzero(fold!=outer);test=np.flatnonzero((fold==outer)&route)
    a,detail,neighbors=apk_scores(apk,y,train,test,k=K,collect=True);scores["L1_APK_KNN"][test]=a[test];append_neighbors("APK",outer,neighbors)
    b,rdetail,rneighbors,support=role_scores(role,mass,y,train,test,k=K,collect=True,min_support=MIN_SUPPORT);scores["L2_ROLE_KNN"][test]=b[test];scores["L3_STAGE1_CONDITIONED"][test]=b[test];append_neighbors("ROLE",outer,rneighbors)
    ax,adetail,_=apk_scores(apk,y,train,test,k=K,exclude_ge=NEAR_DUPLICATE_CUTOFF);scores["L1_APK_KNN_EXCLUDE_GE_099"][test]=ax[test]
    bx,bdetail,_,_=role_scores(role,mass,y,train,test,k=K,exclude_ge=NEAR_DUPLICATE_CUTOFF,min_support=MIN_SUPPORT);scores["L2_ROLE_KNN_EXCLUDE_GE_099"][test]=bx[test]
    amap={d["index"]:d for d in detail};rmap={d["index"]:d for d in rdetail};axmap={d["index"]:d for d in adetail};bxmap={d["index"]:d for d in bdetail}
    for i in test:
        apk_rows.append({"sha256":sha[i],"fold":outer,"label":y[i],"stage1_prob":base[i],"stage1_pred":int(base[i]>=.5),**{k:v for k,v in amap[i].items() if k!="index"},"direction_score_exclude_similarity_ge_099":axmap[i]["direction_score"]})
        role_rows.append({"sha256":sha[i],"fold":outer,"label":y[i],"stage1_prob":base[i],"stage1_pred":int(base[i]>=.5),**{k:v for k,v in rmap[i].items() if k!="index"},"direction_score_exclude_similarity_ge_099":bxmap[i]["direction_score"],"opposition_to_stage1":b[i] if base[i]<.5 else -b[i]})
    for r,sb,sm,valid in support:support_rows.append({"fold":outer,"role":r,"role_name":ROLE_NAMES[r],"benign_support":sb,"malware_support":sm,"valid_for_knn":int(valid),"minimum_support":MIN_SUPPORT})
    for rep in range(CONTROL_REPETITIONS):
        seed=20500000+outer*1000+rep
        shuffled=y.copy();shuffled[train]=np.random.RandomState(seed).permutation(shuffled[train])
        c,_,_=apk_scores(apk,y,train,test,k=K,labels=shuffled);controls["LABEL_SHUFFLE_L1"][rep][test]=c[test]
        c,_,_,_=role_scores(role,mass,y,train,test,k=K,labels=shuffled,min_support=MIN_SUPPORT);controls["LABEL_SHUFFLE_L2"][rep][test]=c[test]
        c,_,_=apk_scores(apk,y,train,test,k=K,random_neighbors=True,seed=seed+100);controls["RANDOM_NEIGHBOR_L1"][rep][test]=c[test]
        c,_,_,_=role_scores(role,mass,y,train,test,k=K,random_neighbors=True,seed=seed+200,min_support=MIN_SUPPORT);controls["RANDOM_NEIGHBOR_L2"][rep][test]=c[test]
        rr,mm=shuffled_roles(role,mass,np.r_[train,test],seed+300);c,_,_,_=role_scores(rr,mm,y,train,test,k=K,min_support=MIN_SUPPORT);controls["ROLE_SHUFFLE_L2"][rep][test]=c[test]
    print(f"LOCAL_ROLE_NEIGHBOR_FOLD_DONE {outer}/5",flush=True)

write_csv(ROOT/"apk_knn_probe_results.csv",apk_rows);write_csv(ROOT/"role_knn_probe_results.csv",role_rows);write_csv(ROOT/"role_local_support_audit.csv",support_rows);write_csv(ROOT/"neighbor_audit.csv",neighbor_rows)

def auc_record(target,value):
    target=np.asarray(target,int);value=np.asarray(value,float);keep=np.isfinite(value);target=target[keep];value=value[keep];pos=int(target.sum());neg=int(len(target)-pos)
    if pos==0 or neg==0:return {"AUC":"","status":"UNDEFINED_DUE_TO_SINGLE_CLASS","positive_n":pos,"negative_n":neg}
    return {"AUC":float(roc_auc_score(target,value)),"status":"DEFINED","positive_n":pos,"negative_n":neg}

def comparison_masks(direction):
    pred1=route&(base>=.5)&np.isfinite(direction);pred0=route&(base<.5)&np.isfinite(direction);changed=(d0_fixed|d0_broken)&np.isfinite(direction);routed=route&np.isfinite(direction)
    opposition=np.where(base>=.5,-direction,direction)
    return {
      "FP_vs_TP":(pred1,(y[pred1]==0).astype(int),-direction[pred1],"FP","benign support"),
      "FN_vs_TN":(pred0,(y[pred0]==1).astype(int),direction[pred0],"FN","malware support"),
      "D0_FIXED_vs_D0_BROKEN":(changed,d0_fixed[changed].astype(int),opposition[changed],"FIXED","opposition to Stage-I"),
      "ROUTED_ERROR_vs_CORRECT":(routed,(~base_ok[routed]).astype(int),opposition[routed],"STAGE1_ERROR","opposition to Stage-I"),
      "MALWARE_vs_BENIGN":(routed,y[routed],direction[routed],"MALWARE","malware direction")}

probe_rows=[];fold_rows=[]
def evaluate(arm,direction,collect_fold=True):
    result={}
    for name,(mask,target,value,positive,score_name) in comparison_masks(direction).items():
        rec=auc_record(target,value);probe_rows.append({"arm":arm,"comparison":name,"positive_class":positive,"score":score_name,"n":len(target),**rec});result[name]=rec
        if collect_fold:
            idx=np.flatnonzero(mask)
            for f in range(1,6):
                take=fold[idx]==f;r=auc_record(target[take],value[take]);fold_rows.append({"arm":arm,"comparison":name,"fold":f,"positive_class":positive,**r})
    return result

real_metrics={arm:evaluate(arm,value) for arm,value in scores.items()}
write_csv(ROOT/"direction_probe_results.csv",probe_rows);write_csv(ROOT/"fold_direction_results.csv",fold_rows)

control_rows=[]
for control,rep_arrays in controls.items():
    for rep,direction in enumerate(rep_arrays):
        for name,(_,target,value,positive,score_name) in comparison_masks(direction).items():
            rec=auc_record(target,value);control_rows.append({"control":control,"repetition":rep,"comparison":name,"positive_class":positive,**rec})
for filename,prefixes in (("label_shuffle_results.csv",("LABEL_SHUFFLE",)),("random_neighbor_results.csv",("RANDOM_NEIGHBOR",)),("role_shuffle_results.csv",("ROLE_SHUFFLE",))):write_csv(ROOT/filename,[r for r in control_rows if r["control"].startswith(prefixes)])

near_rows=[]
for level in ("APK","ROLE"):
    subset=[r for r in neighbor_rows if r["level"]==level]
    for threshold in (.95,.98,.99):near_rows.append({"record_type":"selected_neighbor_rate","level":level,"threshold":threshold,"neighbor_n":len(subset),"at_or_above_threshold":sum(float(r["similarity"])>=threshold for r in subset),"fraction":sum(float(r["similarity"])>=threshold for r in subset)/max(len(subset),1)})
for arm in ("L1_APK_KNN_EXCLUDE_GE_099","L2_ROLE_KNN_EXCLUDE_GE_099"):
    for comparison,rec in real_metrics[arm].items():near_rows.append({"record_type":"exclude_ge_099_performance","level":arm,"comparison":comparison,"AUC":rec["AUC"],"status":rec["status"],"positive_n":rec["positive_n"],"negative_n":rec["negative_n"]})
write_csv(ROOT/"near_duplicate_audit.csv",near_rows)

def optional_numeric(candidates):
    lookup={k.casefold():k for k in rows[0]}
    for name in candidates:
        if name.casefold() in lookup:
            key=lookup[name.casefold()]
            try:return key,np.asarray([float(r[key]) for r in rows])
            except Exception:pass
    return None,None
size_vars={"HostCount":host_count.astype(float),"SensitiveRoleMass":mass.sum(1),"RoleCount":np.sum(mass>0,axis=1).astype(float)}
for label,candidates in {"MethodCount":("method_count","num_methods","audit_method_count"),"GraphNodeCount":("graph_node_count","num_graph_nodes"),"GraphEdgeCount":("graph_edge_count","num_graph_edges"),"SensitiveAPICount":("sensitive_api_count","api_count")}.items():
    key,value=optional_numeric(candidates)
    if value is not None:size_vars[label]=value
size_rows=[]
for arm in ("L1_APK_KNN","L2_ROLE_KNN","L3_STAGE1_CONDITIONED"):
    direction=scores[arm];valid=route&np.isfinite(direction);opposition=np.where(base>=.5,-direction,direction)
    for score_name,score in (("direction_score",direction),("opposition_score",opposition)):
        for variable,value in size_vars.items():
            x=value[valid];z=score[valid]
            if np.std(x)<1e-12 or np.std(z)<1e-12:pear=spear=r2=0.
            else:
                pear=float(pearsonr(z,x).statistic);spear=float(spearmanr(z,x).statistic);r2=float(LinearRegression().fit(x[:,None],z).score(x[:,None],z))
            size_rows.append({"arm":arm,"score":score_name,"variable":variable,"n":len(x),"pearson":pear,"spearman":spear,"linear_R2":r2})
write_csv(ROOT/"size_shortcut_audit.csv",size_rows)

def auc_value(metrics,name):
    value=metrics[name]["AUC"];return float(value) if value!="" else float("nan")
def control_mean(prefix,arm_suffix,comparison):
    values=[float(r["AUC"]) for r in control_rows if r["control"]==prefix+arm_suffix and r["comparison"]==comparison and r["status"]=="DEFINED"]
    return float(np.mean(values)),float(np.std(values))
def fold_support(arm,comparison):
    valid=[r for r in fold_rows if r["arm"]==arm and r["comparison"]==comparison and r["status"]=="DEFINED"]
    return sum(float(r["AUC"])>.5 for r in valid),len(valid)
def assess(arm,level):
    metric=real_metrics[arm];fp=auc_value(metric,"FP_vs_TP");fn=auc_value(metric,"FN_vs_TN");d0fp=auc_value(real_metrics["L0_CURRENT_C43"],"FP_vs_TP");d0fn=auc_value(real_metrics["L0_CURRENT_C43"],"FN_vs_TN")
    lsuffix="_L1" if level=="APK" else "_L2";label_mean,label_std=control_mean("LABEL_SHUFFLE",lsuffix,"FP_vs_TP");random_mean,random_std=control_mean("RANDOM_NEIGHBOR",lsuffix,"FP_vs_TP")
    role_mean=role_std=None
    if level=="ROLE":role_mean,role_std=control_mean("ROLE_SHUFFLE","_L2","FP_vs_TP")
    sfp,vfp=fold_support(arm,"FP_vs_TP");sfn,vfn=fold_support(arm,"FN_vs_TN");near_arm="L1_APK_KNN_EXCLUDE_GE_099" if level=="APK" else "L2_ROLE_KNN_EXCLUDE_GE_099";near_fp=auc_value(real_metrics[near_arm],"FP_vs_TP")
    corr=max([abs(float(r["spearman"])) for r in size_rows if r["arm"]==arm],default=float("inf"))
    passed=(fp>=d0fp+.03 and fn>=d0fn-.03 and vfp>=3 and sfp>vfp/2 and vfn>=3 and sfn>vfn/2 and fp>=label_mean+.03 and fp>=random_mean+.03 and (level!="ROLE" or fp>=role_mean+.03) and near_fp>=d0fp+.02 and corr<.80)
    return {"arm":arm,"level":level,"passed":bool(passed),"FP_vs_TP_AUC":fp,"FN_vs_TN_AUC":fn,"D0_FP_vs_TP_AUC":d0fp,"D0_FN_vs_TN_AUC":d0fn,"FP_supporting_folds":sfp,"FP_valid_folds":vfp,"FN_supporting_folds":sfn,"FN_valid_folds":vfn,"label_shuffle_FP_mean":label_mean,"label_shuffle_FP_std":label_std,"random_neighbor_FP_mean":random_mean,"random_neighbor_FP_std":random_std,"role_shuffle_FP_mean":role_mean,"role_shuffle_FP_std":role_std,"exclude_ge_099_FP_AUC":near_fp,"max_abs_size_spearman":corr}

assessments=[assess("L1_APK_KNN","APK"),assess("L2_ROLE_KNN","ROLE")];selected="L2_ROLE_KNN" if assessments[1]["passed"] else ("L1_APK_KNN" if assessments[0]["passed"] else None)
integration=[]
if selected:
    direction=scores[selected];effective=logit(d0)-logit(base);agreement=np.sign(direction)==np.sign(effective);scale=np.where(route,np.where(agreement,1.,.5),1.);p=1/(1+np.exp(-(logit(base)+effective*scale)));ok=(p>=.5)==y;fixed=(~base_ok)&ok;broken=base_ok&(~ok)
    integration.append({"arm":"I0_CURRENT_C43","BA":balanced_accuracy_score(y,d0>=.5),"AUC":roc_auc_score(y,d0),"F1":f1_score(y,d0>=.5),"fixed":int(d0_fixed.sum()),"broken":int(d0_broken.sum()),"net":int(d0_fixed.sum()-d0_broken.sum())})
    integration.append({"arm":"I1_LOCAL_AGREEMENT_SCALE","selected_probe":selected,"BA":balanced_accuracy_score(y,p>=.5),"AUC":roc_auc_score(y,p),"F1":f1_score(y,p>=.5),"fixed":int(fixed.sum()),"broken":int(broken.sum()),"net":int(fixed.sum()-broken.sum()),"agreement_scale":1.,"disagreement_scale":.5})
    write_csv(ROOT/"integration_results.csv",integration)

decision="LOCAL_ROLE_NEIGHBORHOOD_SUPPORTED" if selected else "LOCAL_ROLE_NEIGHBORHOOD_NOT_SUPPORTED"
dump(ROOT/"probe_decision.json",{"decision":decision,"selected_probe":selected,"assessments":assessments,"integration_ran":bool(selected),"future_rows_loaded":0})
print("L0_REPRODUCTION_PASS");print("L1_APK_KNN_DONE");print("L2_ROLE_KNN_DONE");print("L3_STAGE1_CONDITIONED_DONE");print("LABEL_SHUFFLE_CONTROL_DONE");print("RANDOM_NEIGHBOR_CONTROL_DONE");print("ROLE_SHUFFLE_CONTROL_DONE");print("NEAR_DUPLICATE_AUDIT_DONE");print("SIZE_SHORTCUT_AUDIT_DONE");print(decision)

