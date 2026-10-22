#!/usr/bin/env python3
from __future__ import annotations
import sys,json,itertools
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,f1_score
from common import ROOT,BASE,CACHE,METHOD,ROLE_NAMES,read_csv,write_csv,dump,norm,logit,unit
from role_direction import make_apk_repr,prototypes,scores,shuffled_roles,MIN_SUPPORT

GAMMA=(.5,1.,2.)
rows=read_csv(BASE);sha=np.asarray([norm(r["sha256"]) for r in rows]);y=np.asarray([int(r["label"]) for r in rows]);fold=np.asarray([int(r["fold"]) for r in rows]);base=np.asarray([float(r["Global24__prob"]) for r in rows]);route=np.asarray([int(r["A25__routed"]) for r in rows],bool);d0=np.asarray([float(r[METHOD+"__prob"]) for r in rows]);d0score=np.asarray([float(r[METHOD+"__bounded_delta"]) for r in rows])
with np.load(CACHE,allow_pickle=False) as z:
    val=np.asarray(z["host_values"],float);off=np.asarray(z["host_offsets"],int)
role=np.zeros((799,31,43));mass=np.zeros((799,31));globalv=np.zeros((799,43))
for i in range(799):role[i],mass[i],globalv[i]=make_apk_repr(val[off[i]:off[i+1]])
if not np.isfinite(role).all():raise RuntimeError("NaN/Inf role representation")

def nested_context(outer):
    p=Path("artifacts/care-role-reliability-gate-v1/nested")/f"outer_{outer}.npz"
    train=np.flatnonzero(fold!=outer)
    if p.exists():
        with np.load(p,allow_pickle=False) as z:idx=z["index"].astype(int);bp=z["base"];rr=z["route"].astype(bool)
        if set(idx.tolist())!=set(train.tolist()):raise RuntimeError("nested context cohort mismatch")
        order={v:i for i,v in enumerate(idx)};return np.asarray([bp[order[i]] for i in train]),np.asarray([rr[order[i]] for i in train]),"reused_strict_nested_gate_context"
    # Label-safe fallback: fit the formal P31 committee on the other three original folds.
    S=Path(__file__).resolve().parent.parent/"care-p31-stage1-rf-selection-source-v1"
    if str(S) not in sys.path:sys.path.insert(0,str(S))
    from stage1_core import fit_calibrated_committee,mean_opposition
    with np.load("artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz",allow_pickle=False) as z:ps=np.asarray([norm(x) for x in z["sha256"]]);px=np.asarray(z["role_prevalence"],float)
    pos={s:i for i,s in enumerate(ps)};p31=px[[pos[s] for s in sha]];bp=np.full(799,np.nan);rr=np.zeros(799,bool)
    for held in sorted(set(fold.tolist())-{outer}):
        fit=np.flatnonzero((fold!=outer)&(fold!=held));va=np.flatnonzero(fold==held);members=fit_calibrated_committee(p31[fit],y[fit],p31[va],20280000+outer*1000+held*10,3,4);bp[va]=members[:,1];sc=mean_opposition(members,bp[va]);n=int(round(len(va)*.25));o=np.lexsort((sha[va],-sc));rr[va[o[:n]]]=True
    return bp[train],rr[train],"rebuilt_strict_inner_oof"

def inner_direction_scores(outer,kind):
    train=np.flatnonzero(fold!=outer);out=np.full(799,np.nan)
    for held in sorted(set(fold.tolist())-{outer}):
        fit=np.flatnonzero((fold!=outer)&(fold!=held));va=np.flatnonzero(fold==held);st=prototypes(role,globalv,mass,y,fit);a,b,_,_=scores(st,role,globalv,mass,va);out[va]=a if kind=="D1" else b
    return out[train]

def apply(basep,score,routed,g0,g1):
    gamma=np.where(basep>=.5,g1,g0);delta=2.25*np.tanh(gamma*score);return np.where(routed,1/(1+np.exp(-(logit(basep)+delta))),basep)
def tune(basep,routed,score,labels,pairs):
    best=None
    for g0,g1 in pairs:
        p=apply(basep,score,routed,g0,g1);bo=(basep>=.5)==labels;ok=(p>=.5)==labels;fx=int(np.sum(~bo&ok));br=int(np.sum(bo&~ok));ba=balanced_accuracy_score(labels,p>=.5);key=(fx-br,-br,ba,-g0-g1)
        if best is None or key>best[0]:best=(key,g0,g1,{"inner_fixed":fx,"inner_broken":br,"inner_net":fx-br,"inner_BA":ba})
    return best[1],best[2],best[3]

pred={k:np.full(799,np.nan) for k in ("D1_GLOBAL","D2_PER_ROLE","D3_STAGE_CONDITIONED","C1_CLASS_SHUFFLE","C2_ROLE_SHUFFLE")};score_all={k:np.full(799,np.nan) for k in pred};coverage=np.zeros(799);support_rows=[];sep_rows=[];tune_rows=[]
for outer in range(1,6):
    tr=np.flatnonzero(fold!=outer);te=np.flatnonzero(fold==outer)
    cp=ROOT/"fold_checkpoints"/f"fold_{outer}.npz";cm=ROOT/"fold_checkpoints"/f"fold_{outer}.json"
    if cp.exists() and cm.exists():
        with np.load(cp,allow_pickle=False) as z:
            if not np.array_equal(z["test_index"],te):raise RuntimeError("checkpoint fold membership mismatch")
            for k in pred:pred[k][te]=z["pred__"+k];score_all[k][te]=z["score__"+k]
            coverage[te]=z["coverage"]
        meta=json.loads(cm.read_text());support_rows.extend(meta["support_rows"]);sep_rows.extend(meta["separation_rows"]);tune_rows.extend(meta["tuning_rows"])
        print("CLASS_ROLE_DIRECTION_FOLD_RESUME",outer,flush=True);continue
    state=prototypes(role,globalv,mass,y,tr);s1,s2,cov,per=scores(state,role,globalv,mass,te);coverage[te]=cov
    ib,ir,ctx=nested_context(outer);is1=inner_direction_scores(outer,"D1");is2=inner_direction_scores(outer,"D2")
    g10,g11,a1=tune(ib,ir,is1,y[tr],[(g,g) for g in GAMMA]);g20,g21,a2=tune(ib,ir,is2,y[tr],[(g,g) for g in GAMMA]);g30,g31,a3=tune(ib,ir,is2,y[tr],list(itertools.product(GAMMA,GAMMA)))
    pred["D1_GLOBAL"][te]=apply(base[te],s1,route[te],g10,g11);pred["D2_PER_ROLE"][te]=apply(base[te],s2,route[te],g20,g21);pred["D3_STAGE_CONDITIONED"][te]=apply(base[te],s2,route[te],g30,g31);score_all["D1_GLOBAL"][te]=s1;score_all["D2_PER_ROLE"][te]=s2;score_all["D3_STAGE_CONDITIONED"][te]=s2
    tune_rows += [{"fold":outer,"arm":"D1_GLOBAL","gamma_pred_benign":g10,"gamma_pred_malware":g11,"context":ctx,**a1},{"fold":outer,"arm":"D2_PER_ROLE","gamma_pred_benign":g20,"gamma_pred_malware":g21,"context":ctx,**a2},{"fold":outer,"arm":"D3_STAGE_CONDITIONED","gamma_pred_benign":g30,"gamma_pred_malware":g31,"context":ctx,**a3}]
    # Matched null 1: permute class assignments only inside outer training.
    ys=y.copy();rng=np.random.RandomState(20300000+outer);ys[tr]=rng.permutation(ys[tr]);cs=prototypes(role,globalv,mass,ys,tr);_,ss,_,_=scores(cs,role,globalv,mass,te);score_all["C1_CLASS_SHUFFLE"][te]=ss;pred["C1_CLASS_SHUFFLE"][te]=apply(base[te],ss,route[te],g20,g21)
    # Matched null 2: independently permute Role identity within every APK.
    rs,ms=shuffled_roles(role,mass,np.arange(799),20400000+outer);rstate=prototypes(rs,globalv,ms,y,tr);_,sr,_,_=scores(rstate,rs,globalv,ms,te);score_all["C2_ROLE_SHUFFLE"][te]=sr;pred["C2_ROLE_SHUFFLE"][te]=apply(base[te],sr,route[te],g20,g21)
    for r in range(31):
        valid=bool(state["valid"][r]);cos=float(state["pb"][r]@state["pm"][r]) if valid else float("nan");bidx=tr[(y[tr]==0)&(mass[tr,r]>0)];midx=tr[(y[tr]==1)&(mass[tr,r]>0)];ivb=float(np.mean(1-role[bidx,r]@state["pb"][r])) if valid else float("nan");ivm=float(np.mean(1-role[midx,r]@state["pm"][r])) if valid else float("nan");inter=1-cos if valid else float("nan")
        support_rows.append({"fold":outer,"role":r,"role_name":ROLE_NAMES[r],"benign_train_support":int(state["sb"][r]),"malware_train_support":int(state["sm"][r]),"total_support":int(state["sb"][r]+state["sm"][r]),"prototype_valid":int(valid),"benign_norm":float(np.linalg.norm(state["pb"][r])),"malware_norm":float(np.linalg.norm(state["pm"][r])),"prototype_cosine":cos,"cosine_distance":inter})
        sep_rows.append({"fold":outer,"role":r,"role_name":ROLE_NAMES[r],"prototype_valid":int(valid),"inter_class_distance":inter,"benign_intra_variance":ivb,"malware_intra_variance":ivm,"fisher_like_separation":inter/max(ivb+ivm,1e-8) if valid else float("nan")})
    cp.parent.mkdir(parents=True,exist_ok=True)
    local_support=support_rows[-31:];local_sep=sep_rows[-31:];local_tune=tune_rows[-3:]
    np.savez_compressed(cp,test_index=te,coverage=coverage[te],**{"pred__"+k:pred[k][te] for k in pred},**{"score__"+k:score_all[k][te] for k in pred})
    dump(cm,{"fold":outer,"support_rows":local_support,"separation_rows":local_sep,"tuning_rows":local_tune,"future_rows_loaded":0})
    print("CLASS_ROLE_DIRECTION_FOLD_DONE",outer,flush=True)

arms={"D0_CURRENT_C43":d0,**pred};scores_by={"D0_CURRENT_C43":d0score,**score_all};baseok=(base>=.5)==y;d0ok=(d0>=.5)==y;d0fixed=(~baseok)&d0ok;d0broken=baseok&(~d0ok)
def auc_safe(target,value):return float(roc_auc_score(target,value)) if len(set(np.asarray(target).tolist()))>1 else float("nan")
probe=[];agg=[];foldres=[];fb=[];side=[];outrows=[]
for arm,p in arms.items():
    sc=scores_by[arm];ok=(p>=.5)==y;fixed=(~baseok)&ok;broken=baseok&(~ok);m={"BA":balanced_accuracy_score(y,p>=.5),"AUC":roc_auc_score(y,p),"F1":f1_score(y,p>=.5)}
    pred1=route&(base>=.5);pred0=route&(base<.5);fp_auc=auc_safe((y[pred1]==0).astype(int),-sc[pred1]);fn_auc=auc_safe((y[pred0]==1).astype(int),sc[pred0]);changed=d0fixed|d0broken;opp=np.where(base>=.5,-sc,sc);fb_auc=auc_safe(d0fixed[changed].astype(int),opp[changed]);err=route&(~baseok);diracc=float(np.mean(np.where(y[err]==1,sc[err]>0,sc[err]<0)))
    agg.append({"arm":arm,**m,"fixed":int(fixed.sum()),"broken":int(broken.sum()),"net":int(fixed.sum()-broken.sum()),"routed_direction_accuracy":diracc,"FP_vs_TP_AUC":fp_auc,"FN_vs_TN_AUC":fn_auc,"FIXED_vs_BROKEN_AUC":fb_auc})
    probe += [{"arm":arm,"comparison":"FP_vs_TP","positive_class":"FP","score":"benign support = -malware_direction","n":int(pred1.sum()),"AUC":fp_auc},{"arm":arm,"comparison":"FN_vs_TN","positive_class":"FN","score":"malware support","n":int(pred0.sum()),"AUC":fn_auc},{"arm":arm,"comparison":"D0_FIXED_vs_D0_BROKEN","positive_class":"FIXED","score":"opposition to Stage-I prediction","n":int(changed.sum()),"AUC":fb_auc}]
    fb.append({"arm":arm,"fixed_n":int(d0fixed.sum()),"broken_n":int(d0broken.sum()),"fixed_score_mean":float(np.mean(opp[d0fixed])),"fixed_score_std":float(np.std(opp[d0fixed])),"fixed_score_median":float(np.median(opp[d0fixed])),"broken_score_mean":float(np.mean(opp[d0broken])),"broken_score_std":float(np.std(opp[d0broken])),"broken_score_median":float(np.median(opp[d0broken])),"AUC":fb_auc})
    side.append({"arm":arm,"baseline_FP":int(np.sum(route&(base>=.5)&(y==0))),"fixed_FP":int(np.sum(fixed&(y==0))),"broken_TN_to_FP":int(np.sum(broken&(y==0))),"FP_side_net":int(np.sum(fixed&(y==0))-np.sum(broken&(y==0))),"baseline_FN":int(np.sum(route&(base<.5)&(y==1))),"fixed_FN":int(np.sum(fixed&(y==1))),"broken_TP_to_FN":int(np.sum(broken&(y==1))),"FN_side_net":int(np.sum(fixed&(y==1))-np.sum(broken&(y==1)))})
    for f in range(1,6):
        z=fold==f;fo=(p[z]>=.5)==y[z];bo=baseok[z];foldres.append({"arm":arm,"fold":f,"BA":balanced_accuracy_score(y[z],p[z]>=.5),"AUC":roc_auc_score(y[z],p[z]),"F1":f1_score(y[z],p[z]>=.5),"fixed":int(np.sum(~bo&fo)),"broken":int(np.sum(bo&~fo)),"net":int(np.sum(~bo&fo)-np.sum(bo&~fo))})
    for i in range(799):outrows.append({"sha256":sha[i],"fold":fold[i],"label":y[i],"route":int(route[i]),"stage1_prob":base[i],"arm":arm,"direction_score":sc[i],"final_prob":p[i],"correct":int(ok[i]),"fixed":int(fixed[i]),"broken":int(broken[i])})
write_csv(ROOT/"role_prototype_support.csv",support_rows);write_csv(ROOT/"prototype_separation_audit.csv",sep_rows);write_csv(ROOT/"direction_probe_results.csv",probe);write_csv(ROOT/"fixed_broken_direction_audit.csv",fb);write_csv(ROOT/"source_fold_results.csv",foldres);write_csv(ROOT/"source_predictions.csv",outrows);write_csv(ROOT/"fp_fn_direction_audit.csv",side);write_csv(ROOT/"shuffle_control_results.csv",[r for r in agg if r["arm"] in {"D2_PER_ROLE","C1_CLASS_SHUFFLE","C2_ROLE_SHUFFLE"}]);write_csv(ROOT/"source_aggregate.csv",agg);write_csv(ROOT/"inner_gamma_selection.csv",tune_rows)
print("D0_DONE");print("D1_DONE");print("D2_DONE");print("D3_DONE");print("CLASS_SHUFFLE_CONTROL_DONE");print("PROTOTYPE_SUPPORT_AUDIT_DONE");print("FP_FN_DIRECTION_AUDIT_DONE")

