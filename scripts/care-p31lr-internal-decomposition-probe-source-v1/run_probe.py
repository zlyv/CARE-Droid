#!/usr/bin/env python3
from __future__ import annotations
import csv,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent.parent;SCRIPTS=PROJECT/"scripts"
for p in (HERE,SCRIPTS,SCRIPTS/"care-stage2only-anomaly-v1/vendor",SCRIPTS/"care-litelocal-p31lr-c43-prov6-source-v1",SCRIPTS/"care-di25-graphglobal-path-source-v1",SCRIPTS/"care-rolegraph-conditioned-v4-script",SCRIPTS/"care-rgcondattn-dedup-bounded-source-v1",SCRIPTS/"care-malben-host-v2"):
    if str(p.resolve()) not in sys.path:sys.path.insert(0,str(p.resolve()))
import run_care_rolegraph_conditioned_dualstream_v4 as v4
import prov6_background as pb
from decomposition_core import *

ROOT=Path("artifacts/care-p31lr-internal-decomposition-probe-source-v1")
UP=Path("artifacts/care-litelocal-p31lr-c43-prov6-source-v1")
REF=UP/"preflight/lr_reference/source_predictions.csv";PRED=UP/"arms/P0_ZERO6/source_predictions.csv";CACHE=UP/"cache/P0_ZERO6.npz"
METHOD="P31LR__Opp25__C43Prov6Zero__Bounded"
ARGS=SimpleNamespace(shared_prototypes=32,benign_prototypes=24,prototype_regions_per_apk=16,internal_background_coverage=.70)

def read(p):
    with p.open(encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def write(p,rows):
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf-8",newline="") as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
def logit(p):
    p=np.clip(np.asarray(p,float),1e-8,1-1e-8);return np.log(p/(1-p))

def load_cache(ids):
    with np.load(CACHE,allow_pickle=False) as z:
        zids=[str(v).strip().casefold() for v in z["sha256"]];values=np.asarray(z["host_values"],np.float32);off=np.asarray(z["host_offsets"],int);real=np.asarray(z["real_host_count"],int)
    index={s:i for i,s in enumerate(zids)};bags=[]
    for sha in ids:
        i=index[sha];lo=int(off[i]);n=int(real[i]);bags.append(values[lo:lo+max(n,1)])
    return bags,real[[index[s] for s in ids]]

def fit_transform(train_bags,labels,target_bags,fold_seed,return_hosts=False):
    raw_train,train_roles,_=pb.split_bags(train_bags);raw_target,target_roles,_=pb.split_bags(target_bags)
    mean,std=v4.v3.care_v1.fit_equal_apk_scaler(raw_train,np.arange(len(raw_train),dtype=np.int64))
    scaled_train=pb.weighted(raw_train,mean,std);scaled_target=pb.weighted(raw_target,mean,std)
    models,role_audit=fit_benign_role_models(scaled_train,train_roles,labels)
    refs=fit_percentile_references(scaled_train,train_roles,labels,models,k=3)
    common=[np.zeros(len(x),float) for x in scaled_target];current=[np.zeros(len(x),float) for x in scaled_target]
    seed_audit=[]
    for seed in (0,1,2):
        fullseed=int(fold_seed)*100000+seed;state=v4.v3.v2.fit_prototype_state(scaled_train,np.asarray(labels,int),ARGS,fullseed);sp,bp=pb.profiles(scaled_train,train_roles,labels,state,v4.v3.v2)
        for j,(x,w) in enumerate(zip(scaled_target,target_roles)):
            columns=pb.columns(x,w,state,sp,bp,ARGS,v4.v3.v2);common[j]+=columns[:,2]/3.;current[j]+=columns[:,1]/3.
        seed_audit.append({"seed":fullseed,"shared_n":int(state["shared_k"]),"benign_n":int(state["benign_k"])})
    rows=[];hosts=[]
    for j,(x,w) in enumerate(zip(scaled_target,target_roles)):
        raw_d,q,ca,cr,pr=host_axes(x,w,models,refs,k=3)
        all_d=decompose(q,ca);role_d=decompose(q,cr);shared_d=decompose(q,cr,np.clip(common[j],0,1))
        row={"ALL_B":all_d["B"],"ALL_M_EXCEPTION":all_d["M_exception"],"ALL_M_PERVASIVE":all_d["M_pervasive"],"RULE_ALL":all_d["score"],"ROLE_B":role_d["B"],"ROLE_M_EXCEPTION":role_d["M_exception"],"ROLE_M_PERVASIVE":role_d["M_pervasive"],"RULE_ROLE":role_d["score"],"SHARED_B":shared_d["B"],"SHARED_M_EXCEPTION":shared_d["M_exception"],"SHARED_M_PERVASIVE":shared_d["M_pervasive"],"RULE_ROLE_SHARED":shared_d["score"],"mean_q_ben":float(np.mean(q)),"mean_support_all":float(np.mean(ca)),"mean_support_role":float(np.mean(cr)),"mean_shared_commonness":float(np.mean(common[j])),"mean_current_internal":float(np.mean(current[j])),"host_n":len(x),"unsupported_role_host_n":int(sum(models[int(r)]["fallback"] for r in pr))}
        rows.append(row)
        if return_hosts:
            for h in range(len(x)):hosts.append({"target_index":j,"host_index":h,"primary_role":int(pr[h]),"raw_benign_distance":float(raw_d[h]),"q_ben":float(q[h]),"support_all":float(ca[h]),"support_role":float(cr[h]),"shared_commonness":float(common[j][h]),"current_internal":float(current[j][h]),"benign_consensus_role":float(cr[h]*(1-q[h])),"isolated_exception_role":float(q[h]*(1-cr[h])),"pervasive_exception_role":float(q[h]*cr[h])})
    audit={"scaler_zero_std_n":int(np.sum(std==1)),"benign_role_models":role_audit,"seeds":seed_audit}
    return rows,hosts,audit

def vector(rows,prefix):
    return np.asarray([[r[prefix+"_B"],r[prefix+"_M_EXCEPTION"],r[prefix+"_M_PERVASIVE"]] for r in rows],float)

def main():
    ref=sorted(read(REF),key=lambda r:r["sha256"]);pred=sorted(read(PRED),key=lambda r:r["sha256"]);ids=[r["sha256"].strip().casefold() for r in ref]
    if ids!=[r["sha256"].strip().casefold() for r in pred]:raise RuntimeError("BLOCKED_BY_CONTRACT: cohort mismatch")
    y=np.asarray([int(r["label"]) for r in ref]);folds=np.asarray([int(r["fold"]) for r in ref]);p=np.asarray([float(r["P31_LR__prob"]) for r in ref]);route=np.asarray([int(r["A25__routed"])==1 for r in ref]);stage=(p>=.5).astype(int);error=route&(stage!=y);truth=np.where(y==1,1,-1)
    expert=np.asarray([float(r[METHOD+"__expert_prob"]) for r in pred]);e0=logit(expert)-logit(p)
    bags,real=load_cache(ids);apk_rows=[];host_rows=[];fold_audits=[]
    methods=("E0_CURRENT_DELTA","RULE_ALL","RULE_ROLE","RULE_ROLE_SHARED","LOGIT_ALL","LOGIT_ROLE","LOGIT_ROLE_SHARED")
    for fold in range(1,6):
        train=np.flatnonzero(folds!=fold);val=np.flatnonzero((folds==fold)&route)
        train_features=[None]*len(train)
        inner=StratifiedKFold(4,shuffle=True,random_state=9100+fold)
        for inner_id,(fit_pos,hold_pos) in enumerate(inner.split(train,y[train]),1):
            fit_idx=train[fit_pos];hold_idx=train[hold_pos]
            part,_,audit=fit_transform([bags[i] for i in fit_idx],y[fit_idx],[bags[i] for i in hold_idx],fold*10+inner_id)
            for pos,item in zip(hold_pos,part):train_features[int(pos)]=item
            fold_audits.append({"outer_fold":fold,"phase":"inner_oof","inner_fold":inner_id,"fit_n":len(fit_idx),"holdout_n":len(hold_idx),"holdout_labels_used_by_background":0,"scaler_zero_std_n":audit["scaler_zero_std_n"]})
        if any(x is None for x in train_features):raise RuntimeError("BLOCKED_BY_CONTRACT: incomplete inner OOF decomposition")
        val_features,val_hosts,audit=fit_transform([bags[i] for i in train],y[train],[bags[i] for i in val],fold,True)
        fold_audits.append({"outer_fold":fold,"phase":"outer_validation","inner_fold":0,"fit_n":len(train),"holdout_n":len(val),"holdout_labels_used_by_background":0,"scaler_zero_std_n":audit["scaler_zero_std_n"],"supported_benign_roles":int(sum(not x["global_fallback"] for x in audit["benign_role_models"]))})
        probe_scores={}
        for prefix in ("ALL","ROLE","SHARED"):
            model=Pipeline([("scale",StandardScaler()),("lr",LogisticRegression(C=1,class_weight="balanced",solver="liblinear",max_iter=5000,random_state=7300+fold))])
            model.fit(vector(train_features,prefix),y[train]);probe_scores["LOGIT_"+("ROLE_SHARED" if prefix=="SHARED" else prefix)]=model.decision_function(vector(val_features,prefix))
        for local,i in enumerate(val):
            item={"sha256":ids[i],"fold":fold,"label":int(y[i]),"stage1_prob":float(p[i]),"stage1_pred":int(stage[i]),"route":1,"stage1_error":int(stage[i]!=y[i]),"error_type":"FN" if y[i]==1 and stage[i]==0 else "FP" if y[i]==0 and stage[i]==1 else "CORRECT","truth_sign":int(truth[i]) if stage[i]!=y[i] else 0,"real_host_n":int(real[i]),"E0_CURRENT_DELTA":float(e0[i]),**val_features[local]}
            for name,values in probe_scores.items():item[name]=float(values[local])
            apk_rows.append(item)
        for h in val_hosts:
            i=val[int(h.pop("target_index"))];host_rows.append({"sha256":ids[i],"fold":fold,"label":int(y[i]),"stage1_pred":int(stage[i]),"error_type":"FN" if y[i]==1 and stage[i]==0 else "FP" if y[i]==0 and stage[i]==1 else "CORRECT",**h})
        print(f"internal decomposition fold {fold}/5 done",flush=True)
    apk_rows=sorted(apk_rows,key=lambda r:r["sha256"]);errors=[r for r in apk_rows if r["stage1_error"]]
    quality=[];taxonomy=[];perfold=[]
    for m in methods:
        aligned=np.asarray([float(r[m])*int(r["truth_sign"]) for r in errors]);fn=np.asarray([r["error_type"]=="FN" for r in errors]);direction=np.where(aligned>1e-12,1,np.where(aligned<-1e-12,-1,0));fold_acc=[]
        for f in range(1,6):
            sub=[r for r in errors if r["fold"]==f];fold_acc.append(float(np.mean([float(r[m])*int(r["truth_sign"])>1e-12 for r in sub])))
        quality.append({"Method":m,"Overall_direction_acc":float(np.mean(direction>0)),"FN_direction_acc":float(np.mean(direction[fn]>0)),"FP_direction_acc":float(np.mean(direction[~fn]>0)),"Mean_aligned":float(np.mean(aligned)),"Median_aligned":float(np.median(aligned)),"FN_mean_aligned":float(np.mean(aligned[fn])),"FP_mean_aligned":float(np.mean(aligned[~fn])),"Fold_std":float(np.std(fold_acc))})
        taxonomy.append({"Method":m,"WrongDirection":int(np.sum(direction<0)),"Ambiguous":int(np.sum(direction==0)),"CorrectDirection":int(np.sum(direction>0))})
    for f in range(1,6):
        sub=[r for r in errors if r["fold"]==f];row={"Fold":f,"Routed_errors":len(sub),"FN":sum(r["error_type"]=="FN" for r in sub),"FP":sum(r["error_type"]=="FP" for r in sub)}
        for m in methods:row[m+"_acc"]=float(np.mean([float(r[m])*int(r["truth_sign"])>1e-12 for r in sub]))
        perfold.append(row)
    q={r["Method"]:r for r in quality};base=q["E0_CURRENT_DELTA"];eligible=[]
    for m in methods[1:]:
        noninferior=sum(next(r for r in perfold if r["Fold"]==f)[m+"_acc"]>=next(r for r in perfold if r["Fold"]==f)["E0_CURRENT_DELTA_acc"] for f in range(1,6));q[m]["Noninferior_folds_vs_E0"]=noninferior
        if q[m]["FN_direction_acc"]>=base["FN_direction_acc"] and q[m]["FP_direction_acc"]>base["FP_direction_acc"] and q[m]["Overall_direction_acc"]>base["Overall_direction_acc"] and noninferior>=3:eligible.append(m)
    decision=max(eligible,key=lambda m:(q[m]["Overall_direction_acc"],q[m]["FP_direction_acc"],-q[m]["Fold_std"])) if eligible else "NONE"
    # Mechanism diagnostics: role masking effect and unwanted APK-size dependence.
    masking=[]
    for cohort,sub in (("ALL_ROUTED",apk_rows),("ROUTED_ERRORS",errors),("FN",[r for r in errors if r["error_type"]=="FN"]),("FP",[r for r in errors if r["error_type"]=="FP"])):
        a=np.asarray([r["mean_support_all"] for r in sub]);b=np.asarray([r["mean_support_role"] for r in sub]);masking.append({"cohort":cohort,"n":len(sub),"all_host_support_mean":float(a.mean()),"same_role_support_mean":float(b.mean()),"role_minus_all_mean":float((b-a).mean()),"role_lt_all_fraction":float(np.mean(b<a))})
    size=[]
    from scipy.stats import spearmanr
    for name in ("ALL_B","ALL_M_EXCEPTION","ALL_M_PERVASIVE","ROLE_B","ROLE_M_EXCEPTION","ROLE_M_PERVASIVE","SHARED_B","SHARED_M_EXCEPTION","SHARED_M_PERVASIVE"):
        rho=float(spearmanr([r[name] for r in apk_rows],[r["real_host_n"] for r in apk_rows]).statistic);size.append({"feature":name,"rho_real_host_n":0. if not np.isfinite(rho) else rho,"abs_rho":0. if not np.isfinite(rho) else abs(rho)})
    write(ROOT/"direction_quality.csv",quality);write(ROOT/"per_fold_direction.csv",perfold);write(ROOT/"failure_taxonomy.csv",taxonomy);write(ROOT/"routed_apk_decomposition.csv",apk_rows);write(ROOT/"routed_host_decomposition.csv",host_rows);write(ROOT/"fold_audit.csv",fold_audits);write(ROOT/"internal_masking_audit.csv",masking);write(ROOT/"size_association.csv",size)
    protocol={"source_only":True,"source_rows":799,"stage1":"P31 calibrated LR only","router":"LR-anchored MeanOpposition Top25%","routed":200,"routed_errors":99,"FN":int(np.sum(error&(y==1))),"FP":int(np.sum(error&(y==0))),"outer_train_only_backgrounds":True,"outer_train_features_for_linear_probe":"strict inner-OOF decomposition","benign_axis":"role-conditioned empirical percentile","internal_axes":"leave-one-host-out Top3 KNN all-host and same-role","shared_commonness_role":"nonnegative reliability weight only","fixed_rule":"0.5*(M_exception+M_pervasive)-B_consensus","linear_probe_training_invoked":True,"linear_probe_scope":"source outer-train only; diagnostic, not C43","neural_training_invoked":False,"future_rows_loaded":0,"advance_candidate":decision}
    (ROOT/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n",encoding="utf-8")
    lines=["CARE-Droid P31-LR internal benign-consensus decomposition probe source v1","","Strict 2017 OOF; P31-LR and LR-anchored DI25 fixed. No neural training or future row.","B=benign consensus; M_exception=isolated anomaly; M_pervasive=repeated internally-supported anomaly.",""]
    for r in quality:lines.append(f"{r['Method']}: overall={r['Overall_direction_acc']:.6f} FN={r['FN_direction_acc']:.6f} FP={r['FP_direction_acc']:.6f} mean_aligned={r['Mean_aligned']:.6f} median={r['Median_aligned']:.6f} fold_std={r['Fold_std']:.6f}")
    lines += ["",f"DECOMPOSITION_PROBE_ADVANCE_CANDIDATE={decision}","Advance only requires simultaneous FN non-inferiority, FP improvement, overall improvement, and >=3 noninferior folds.","If NONE, do not modify C43 or run a future diagnostic."]
    (ROOT/"REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8");print("P31LR_INTERNAL_DECOMPOSITION_PROBE_DONE");print("\n".join(lines))

if __name__=="__main__":main()
