#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
from common import ROOT,BASE,CACHE,METHOD,read_csv,write_csv,dump,norm,logit,entropy

rows=read_csv(BASE);by={norm(r["sha256"]):r for r in rows}
with np.load(CACHE,allow_pickle=False) as z:
    sha=z["sha256"].astype(str);values=np.asarray(z["host_values"],float);off=np.asarray(z["host_offsets"],int)
if len(sha)<799 or not np.array_equal(np.asarray([norm(r["sha256"]) for r in rows]),np.asarray([norm(x) for x in sha[:799]])):raise RuntimeError("BLOCKED_BY_CONTRACT: cache SHA order mismatch")
diag_names=("critical_score","mean_instance_score","attention_entropy","max_attention","context_entropy","mean_external_distance","mean_internal_distance","mean_shared_commonness","max_local_residual","graph_pool_entropy")
all_rows=[]
for i,r in enumerate(rows):
    x=values[off[i]:off[i+1]]
    if x.ndim!=2 or x.shape[1]<105:raise RuntimeError(f"BLOCKED_BY_CONTRACT: Host width {x.shape}")
    # A0 stored layout is Compact43 | SelfRole31 | Relation31 | sidecars.
    # Role evidence is therefore 43:74, not the first 31 compact coordinates.
    w=np.maximum(x[:,43:74],0);host_mass=w.sum(1);apk=w.sum(0);active=apk>1e-8
    role_p=np.divide(apk,apk.sum(),out=np.zeros_like(apk),where=apk.sum()>0)
    support=(w>1e-8).sum(0);active_support=support[active]
    per_host=np.divide(w,w.sum(1,keepdims=True),out=np.zeros_like(w),where=w.sum(1,keepdims=True)>0)
    relation=x[:,74:105]
    base=float(r["Global24__prob"]);cand=float(r[METHOD+"__prob"]);y=int(r["label"]);b=int(base>=.5);c=int(cand>=.5)
    if b!=y and c==y:outcome="FIXED"
    elif b==y and c!=y:outcome="BROKEN"
    elif b==y and c==y:outcome="UNCHANGED_CORRECT"
    else:outcome="UNCHANGED_WRONG"
    q={"sha256":norm(r["sha256"]),"fold":int(r["fold"]),"label":y,"route":int(r["A25__routed"]),"stage1_prob":base,"candidate_prob":cand,"stage1_pred":b,"candidate_pred":c,"outcome":outcome,"prediction_changed":int(b!=c),"candidate_delta":float(r[METHOD+"__bounded_delta"]),"stage1_margin":abs(float(logit([base])[0])),"stage2_margin":abs(float(logit([cand])[0])),"router_score":float(r["A25__score"]),"host_count":len(x),"active_role_count":int(active.sum()),"active_role_ratio":float(active.mean()),"role_entropy":float(entropy(role_p)/np.log(31)),"max_role_weight":float(role_p.max(initial=0)),"top3_role_concentration":float(np.sort(role_p)[-3:].sum()),"mean_active_role_host_support":float(active_support.mean()) if len(active_support) else 0.,"max_active_role_host_support_ratio":float(active_support.max()/max(len(x),1)) if len(active_support) else 0.,"dominant_host_role_mass_ratio":float(host_mass.max(initial=0)/max(host_mass.sum(),1e-8)),"role_dispersion":float(np.mean(np.var(per_host[:,active],axis=0))) if np.any(active) else 0.,"relation_nonzero_host_ratio":float(np.mean(np.sum(np.abs(relation),axis=1)>1e-8)),"relation_mean_l1":float(np.mean(np.sum(np.abs(relation),axis=1))),"relation_max_l1":float(np.max(np.sum(np.abs(relation),axis=1),initial=0)),"role_occurrence_mass":float(np.sum(w>1e-8))}
    for name in diag_names:q[name]=float(r[METHOD+"__"+name])
    q["abs_candidate_delta"]=abs(q["candidate_delta"])
    if not np.isfinite(np.asarray([v for v in q.values() if isinstance(v,(int,float))],float)).all():raise RuntimeError("NaN/Inf reliability features")
    all_rows.append(q)
out=[r for r in all_rows if r["route"]]
write_csv(ROOT/"all_reliability_features.csv",all_rows)
write_csv(ROOT/"routed_outcome_table.csv",out)
counts={k:sum(r["outcome"]==k for r in out) for k in ("FIXED","BROKEN","UNCHANGED_CORRECT","UNCHANGED_WRONG")}
dump(ROOT/"outcome_summary.json",{"routed":len(out),"counts":counts,"prediction_changed":sum(r["prediction_changed"] for r in out),"future_rows_loaded":0})
print("ROLE_RELIABILITY_OUTCOMES_DONE",counts)

