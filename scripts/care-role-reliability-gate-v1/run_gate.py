#!/usr/bin/env python3
from __future__ import annotations
import json,math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,f1_score
from common import ROOT,BASE,METHOD,read_csv,write_csv,dump,logit,norm

STAGE=("stage1_margin","router_score")
COUNT=("host_count","active_role_count","role_occurrence_mass")
ROLE=("active_role_ratio","role_entropy","max_role_weight","top3_role_concentration","mean_active_role_host_support","max_active_role_host_support_ratio","dominant_host_role_mass_ratio","role_dispersion","relation_nonzero_host_ratio","relation_mean_l1","relation_max_l1","abs_candidate_delta","critical_score","mean_instance_score","attention_entropy","max_attention","context_entropy","mean_external_distance","mean_internal_distance","mean_shared_commonness","max_local_residual","graph_pool_entropy")
DIAG=("bounded_delta","critical_score","mean_instance_score","attention_entropy","max_attention","context_entropy","mean_external_distance","mean_internal_distance","mean_shared_commonness","max_local_residual","graph_pool_entropy")
ARMS={"G1_STAGE1":STAGE,"G2_ROLE":ROLE,"G3_COMBINED":STAGE+ROLE,"C1_G3_SHUFFLE":STAGE+ROLE,"C2_COUNT_ONLY":COUNT}

def model():return Pipeline([("scale",StandardScaler()),("lr",LogisticRegression(C=1.,solver="liblinear",class_weight={0:2.,1:1.},max_iter=5000,random_state=20260930))])
def choose_threshold(x,y,fold):
    pred=np.full(len(y),np.nan)
    for f in sorted(set(fold.tolist())):
        tr=fold!=f;va=fold==f
        if len(set(y[tr].tolist()))<2:continue
        m=model();m.fit(x[tr],y[tr]);pred[va]=m.predict_proba(x[va])[:,1]
    valid=np.isfinite(pred)
    if valid.sum()<6:return .9,{"reason":"insufficient_inner_gate_cv","valid":int(valid.sum())}
    best=None
    for t in (.50,.60,.70,.80,.90):
        accept=pred[valid]>=t;yy=y[valid];fixed=int(np.sum(accept&(yy==1)));broken=int(np.sum(accept&(yy==0)));utility=fixed-2*broken
        row=(utility,fixed,-broken,-t,t,fixed,broken)
        if best is None or row>best:best=row
    return float(best[4]),{"utility_fixed_minus_2broken":best[0],"fixed":best[5],"broken":best[6],"valid":int(valid.sum())}
def metric(y,p):return {"BA":balanced_accuracy_score(y,p>=.5),"AUC":roc_auc_score(y,p),"F1":f1_score(y,p>=.5)}

base_rows=read_csv(BASE);all_rows=read_csv(ROOT/"all_reliability_features.csv")
if len(base_rows)!=799 or len(all_rows)!=799:raise RuntimeError("BLOCKED_BY_CONTRACT: cohort changed")
sha=np.asarray([norm(r["sha256"]) for r in base_rows]);y=np.asarray([int(r["label"]) for r in base_rows]);fold=np.asarray([int(r["fold"]) for r in base_rows]);base=np.asarray([float(r["Global24__prob"]) for r in base_rows]);cand=np.asarray([float(r[METHOD+"__prob"]) for r in base_rows]);route=np.asarray([int(r["A25__routed"]) for r in base_rows],bool)
if not np.array_equal(sha,np.asarray([norm(r["sha256"]) for r in all_rows])):raise RuntimeError("SHA mismatch")
feature={k:np.asarray([float(r[k]) for r in all_rows]) for k in set(STAGE+COUNT+ROLE)}

outer_q={a:np.zeros(799) for a in ARMS};outer_tau={a:np.full(799,np.nan) for a in ARMS};importance=[];fold_gate=[]
for outer in range(1,6):
    path=ROOT/"nested"/f"outer_{outer}.npz"
    if not path.exists():raise RuntimeError("BLOCKED_BY_CONTRACT: missing strict nested replay "+str(path))
    with np.load(path,allow_pickle=False) as z:idx=z["index"].astype(int);bt=z["base"];ct=z["candidate"];rr=z["route"].astype(bool);sc=z["score"];dg=z["diagnostic"]
    if np.any(fold[idx]==outer):raise RuntimeError("BLOCKED_BY_CONTRACT: nested replay includes outer test")
    train_feature={k:feature[k][idx].copy() for k in feature}
    train_feature["stage1_margin"]=np.abs(logit(bt));train_feature["router_score"]=sc
    for j,n in enumerate(DIAG):
        if n=="bounded_delta":train_feature["abs_candidate_delta"]=np.abs(dg[:,j])
        elif n in train_feature:train_feature[n]=dg[:,j]
    test=np.flatnonzero(fold==outer)
    bp=bt>=.5;cp=ct>=.5;changed=rr&(bp!=cp);target=((bp!=y[idx])&(cp==y[idx])).astype(int);sel=np.flatnonzero(changed)
    if len(sel)<6 or len(set(target[sel].tolist()))<2:
        for arm in ARMS:
            outer_q[arm][test]=0.;outer_tau[arm][test]=1.01
            fold_gate.append({"arm":arm,"outer_fold":outer,"nested_changed_n":len(sel),"nested_fixed_n":int(target[sel].sum()),"nested_broken_n":int(len(sel)-target[sel].sum()),"threshold":1.01,"reason":"insufficient nested FIXED/BROKEN support; conservative KEEP"})
        print("ROLE_GATE_OUTER_DEGENERATE_KEEP",outer,len(sel),set(target[sel].tolist()),flush=True)
        continue
    for arm,names in ARMS.items():
        x=np.column_stack([train_feature[n] for n in names])[sel];xt=np.column_stack([feature[n][test] for n in names]);yt=target[sel];ft=fold[idx][sel]
        if arm=="C1_G3_SHUFFLE":
            rng=np.random.RandomState(20260930+outer);role_start=len(STAGE)
            for f in sorted(set(ft.tolist())):
                s=np.flatnonzero(ft==f);perm=rng.permutation(s);x[s,role_start:]=x[perm,role_start:]
        tau,ta=choose_threshold(x,yt,ft);m=model();m.fit(x,yt);q=m.predict_proba(xt)[:,1];outer_q[arm][test]=q;outer_tau[arm][test]=tau
        coef=m.named_steps["lr"].coef_[0]
        for n,c in zip(names,coef):importance.append({"arm":arm,"outer_fold":outer,"feature":n,"standardized_coefficient":c})
        fold_gate.append({"arm":arm,"outer_fold":outer,"nested_changed_n":len(sel),"nested_fixed_n":int(yt.sum()),"nested_broken_n":int(len(yt)-yt.sum()),"threshold":tau,**ta})
    print("ROLE_GATE_OUTER_DONE",outer,flush=True)

arms={"G0_CURRENT":(np.ones(799),np.zeros(799))}|{a:(outer_q[a],outer_tau[a]) for a in ARMS}
predrows=[];agg=[];per=[];side=[]
baseok=(base>=.5)==y;g0ok=(cand>=.5)==y;g0fixed=(~baseok)&g0ok;g0broken=baseok&(~g0ok)
for arm,(q,tau) in arms.items():
    accept=route if arm=="G0_CURRENT" else route&(q>=tau);final=np.where(accept,cand,base);ok=(final>=.5)==y;fixed=(~baseok)&ok;broken=baseok&(~ok);changed=route&((base>=.5)!=(cand>=.5))
    ben=(~baseok)&g0ok;harm=baseok&(~g0ok);accepted_changed=accept&changed
    ba=metric(y,final);benefit_auc=float(roc_auc_score(ben[changed].astype(int),q[changed])) if arm!="G0_CURRENT" and len(set(ben[changed].tolist()))>1 else float("nan")
    direction=[]
    for i in np.flatnonzero(route&(~baseok)):
        truth=1 if y[i]==1 else -1;direction.append((logit([final[i]])[0]-logit([base[i]])[0])*truth>0)
    row={"arm":arm,**ba,"candidate_corrections":int(route.sum()),"accepted_corrections":int(accept.sum()),"fixed":int(fixed.sum()),"broken":int(broken.sum()),"net":int(fixed.sum()-broken.sum()),"fixed_retention":float(fixed.sum()/max(g0fixed.sum(),1)),"broken_suppression":float(1-broken.sum()/max(g0broken.sum(),1)),"benefit_precision":float(np.sum(accepted_changed&ben)/max(np.sum(accepted_changed),1)),"harm_precision":float(np.sum(accepted_changed&harm)/max(np.sum(accepted_changed),1)),"benefit_auc":benefit_auc,"routed_direction_accuracy":float(np.mean(direction)) if direction else float("nan")};agg.append(row)
    for f in range(1,6):
        s=fold==f;per.append({"arm":arm,"fold":f,"fixed":int(fixed[s].sum()),"broken":int(broken[s].sum()),"net":int(fixed[s].sum()-broken[s].sum()),"accepted":int(accept[s].sum()),**metric(y[s],final[s])})
    fixed_fn=fixed&(y==1);fixed_fp=fixed&(y==0);br_tn=broken&(y==0);br_tp=broken&(y==1);side.append({"arm":arm,"fixed_FN":int(fixed_fn.sum()),"broken_TP_to_FN":int(br_tp.sum()),"FN_side_net":int(fixed_fn.sum()-br_tp.sum()),"fixed_FP":int(fixed_fp.sum()),"broken_TN_to_FP":int(br_tn.sum()),"FP_side_net":int(fixed_fp.sum()-br_tn.sum())})
    for i in range(799):predrows.append({"sha256":sha[i],"fold":fold[i],"label":y[i],"arm":arm,"stage1_prob":base[i],"candidate_prob":cand[i],"route":int(route[i]),"gate_probability":q[i],"gate_threshold":tau[i],"accepted":int(accept[i]),"final_prob":final[i],"G0_fixed":int(g0fixed[i]),"G0_broken":int(g0broken[i])})

# Coverage-risk curve uses the honest outer predictions, never a global fitted rank.
curve=[]
for arm in ("G1_STAGE1","G2_ROLE","G3_COMBINED","C1_G3_SHUFFLE","C2_COUNT_ONLY"):
    q=outer_q[arm];idx=np.flatnonzero(route);order=idx[np.argsort(-q[idx],kind="mergesort")]
    for frac in (.1,.2,.3,.4,.5,.6,.7,.8,.9,1.):
        n=int(round(len(order)*frac));acc=np.zeros(799,bool);acc[order[:n]]=True;fx=int(np.sum(acc&g0fixed));br=int(np.sum(acc&g0broken));curve.append({"arm":arm,"accepted_fraction":frac,"accepted_n":n,"fixed":fx,"broken":br,"net":fx-br,"broken_rate_among_changed":br/max(fx+br,1)})
write_csv(ROOT/"gate_fold_results.csv",per);write_csv(ROOT/"gate_predictions.csv",predrows);write_csv(ROOT/"coverage_risk_curve.csv",curve);write_csv(ROOT/"fp_fn_gate_audit.csv",side);write_csv(ROOT/"feature_importance.csv",importance);write_csv(ROOT/"gate_training_audit.csv",fold_gate);write_csv(ROOT/"gate_aggregate.csv",agg)
write_csv(ROOT/"shuffle_control_results.csv",[r for r in agg if r["arm"] in {"G3_COMBINED","C1_G3_SHUFFLE"}]);write_csv(ROOT/"count_only_results.csv",[r for r in agg if r["arm"] in {"G2_ROLE","G3_COMBINED","C2_COUNT_ONLY"}])
print("ROLE_RELIABILITY_GATE_OOF_DONE")

