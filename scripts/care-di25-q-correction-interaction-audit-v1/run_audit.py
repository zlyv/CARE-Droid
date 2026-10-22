#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,f1_score,confusion_matrix
from audit_core import ARMS,clip_logit,interaction

ROOT=Path("artifacts/care-di25-q-correction-interaction-audit-v1")
PATHS={
"H_HOST_ONLY":Path("artifacts/care-di25-qbranch-source-v1/arms/HOST_ONLY_Q/source_predictions.csv"),
"HR_HOST_ROLE":Path("artifacts/care-di25-qbranch-source-v1/arms/STRUCTURE_ONLY_Q/source_predictions.csv"),
"HC_HOST_CONTEXT":Path("artifacts/care-di25-q-drop-role-source-v1/arm/source_predictions.csv"),
"HRC_FULL":Path("artifacts/care-di25-graphglobal-path-source-v1/arms/Q_QUERY_ONLY/source_predictions.csv")}

def read(p):
    with p.open("r",encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def write(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with p.open("w",encoding="utf-8",newline="") as h:w=csv.DictWriter(h,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)
def pcol(rows):
    c=[k for k in rows[0] if k.endswith("__Bounded__prob")]
    if len(c)!=1:raise RuntimeError("probability column mismatch "+repr(c))
    return c[0]
def metrics(y,p,b):
    z=p>=.5;bz=b>=.5;tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1]).ravel();fixed=int(np.sum((z==y)&(bz!=y)));broken=int(np.sum((z!=y)&(bz==y)))
    return {"BA":float(balanced_accuracy_score(y,z)),"AUC":float(roc_auc_score(y,p)),"F1":float(f1_score(y,z)),"TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp),"fixed":fixed,"broken":broken,"net":fixed-broken}

def main():
    tables={a:read(p) for a,p in PATHS.items()};ref=tables["HRC_FULL"];ids=[r["sha256"].lower() for r in ref];maps={a:{r["sha256"].lower():r for r in rs} for a,rs in tables.items()}
    if any(set(m)!=set(ids) for m in maps.values()):raise RuntimeError("BLOCKED_BY_CONTRACT: cohort mismatch")
    y=np.array([int(float(r["label"])) for r in ref]);b=np.array([float(r["Global24__prob"]) for r in ref]);fold=np.array([int(float(r["fold"])) for r in ref]);route=np.array([int(float(r["A25__routed"])) for r in ref],dtype=bool)
    if route.sum()!=200:raise RuntimeError("BLOCKED_BY_CONTRACT: route count changed")
    probs={}
    for a in ARMS:
        rr=[maps[a][i] for i in ids]
        if any(int(float(r["A25__routed"]))!=int(route[j]) for j,r in enumerate(rr)):raise RuntimeError(a+" route mismatch")
        probs[a]=np.array([float(r[pcol(rr)]) for r in rr])
    correct={a:(p>=.5)==y for a,p in probs.items()};base_correct=(b>=.5)==y
    aggregate={a:metrics(y,p,b) for a,p in probs.items()}
    aggregate_rows=[{"arm":a,**aggregate[a]} for a in ARMS]
    interaction_row={"scope":"aggregate"}
    for key in ("BA","AUC","F1","fixed","broken","net"):
        interaction_row["interaction_"+key]=interaction({a:aggregate[a][key] for a in ARMS})
    fold_rows=[];fold_inter=[]
    for f in range(1,6):
        z=fold==f;fm={a:metrics(y[z],probs[a][z],b[z]) for a in ARMS}
        for a in ARMS:fold_rows.append({"fold":f,"arm":a,**fm[a]})
        q={"fold":f}
        for key in ("BA","AUC","F1","fixed","broken","net"):q["interaction_"+key]=interaction({a:fm[a][key] for a in ARMS})
        fold_inter.append(q)
    detail=[]
    for i,sha in enumerate(ids):
        if not route[i]:continue
        row={"sha256":sha,"fold":int(fold[i]),"label":int(y[i]),"stage1_correct":int(base_correct[i]),"stage1_error_type":"none" if base_correct[i] else ("FN" if y[i]==1 else "FP"),"p_stage1":b[i]}
        for a in ARMS:
            row[a+"__prob"]=probs[a][i];row[a+"__correct"]=int(correct[a][i]);row[a+"__delta_logit"]=float(clip_logit([probs[a][i]])[0]-clip_logit([b[i]])[0])
        h,hr,hc,full=(correct[a][i] for a in ARMS)
        row["joint_only_correct"]=int(full and not hr and not hc)
        row["joint_only_wrong"]=int((not full) and hr and hc)
        row["full_prevents_host_harm"]=int((not h) and full)
        row["full_loses_host_benefit"]=int(h and not full)
        row["interaction_correctness"]=(int(full)-int(hr))-(int(hc)-int(h))
        detail.append(row)
    dcount=Counter()
    for r in detail:
        for k in ("joint_only_correct","joint_only_wrong","full_prevents_host_harm","full_loses_host_benefit"):dcount[k]+=int(r[k])
        dcount["interaction_correctness_"+str(r["interaction_correctness"])]+=1
    # Decompose specifically relative to whether Stage I was initially right.
    for stage in (0,1):
        subset=[r for r in detail if r["stage1_correct"]==stage]
        prefix="stage1_correct" if stage else "stage1_error"
        dcount[prefix+"_n"]=len(subset)
        dcount[prefix+"_host_correct"]=sum(r["H_HOST_ONLY__correct"] for r in subset)
        dcount[prefix+"_full_correct"]=sum(r["HRC_FULL__correct"] for r in subset)
        dcount[prefix+"_full_prevents_host_harm"]=sum(r["full_prevents_host_harm"] for r in subset)
        dcount[prefix+"_full_loses_host_benefit"]=sum(r["full_loses_host_benefit"] for r in subset)
    paired=[]
    for a in ("H_HOST_ONLY","HR_HOST_ROLE","HC_HOST_CONTEXT"):
        ca=correct[a];cf=correct["HRC_FULL"];cand=int(np.sum((~cf)&ca));full=int(np.sum(cf&(~ca)));pv=float(binomtest(min(cand,full),cand+full,.5).pvalue) if cand+full else 1.0
        paired.append({"comparison":a+"_vs_HRC_FULL","candidate_only_correct":cand,"full_only_correct":full,"paired_net":cand-full,"exact_p":pv})
    write(ROOT/"aggregate_metrics.csv",aggregate_rows);write(ROOT/"factorial_interaction.csv",[interaction_row]);write(ROOT/"fold_metrics.csv",fold_rows);write(ROOT/"fold_interactions.csv",fold_inter);write(ROOT/"paired_vs_full.csv",paired);write(ROOT/"per_apk_interaction.csv",detail)
    lines=["CARE-Droid Q correction-input 2x2 interaction audit v1","","Read-only strict-2017 OOF audit. No training, inference, route change, Ego, or future row.","H=Host; R=direct RoleToken; C=Global24+Stage-I logit; all variants retain Q-guided localization.",""]
    for a in ARMS:
        m=aggregate[a];lines.append(f"{a}: BA={m['BA']:.6f} AUC={m['AUC']:.6f} F1={m['F1']:.6f} fixed={m['fixed']} broken={m['broken']} net={m['net']}")
    lines += ["",f"R x C interaction: dBA={interaction_row['interaction_BA']:+.6f} dAUC={interaction_row['interaction_AUC']:+.6f} dF1={interaction_row['interaction_F1']:+.6f} dFixed={interaction_row['interaction_fixed']:+.0f} dBroken={interaction_row['interaction_broken']:+.0f} dNet={interaction_row['interaction_net']:+.0f}",""]
    lines += [f"{k}={v}" for k,v in sorted(dcount.items())]
    lines += ["","Interpretation boundary: these are retrained capacity-matched sufficiency arms. The factorial contrast diagnoses non-additivity; it does not prove a causal neural interaction."]
    (ROOT/"REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8");(ROOT/"summary.json").write_text(json.dumps({"aggregate":aggregate,"interaction":interaction_row,"counts":dict(dcount),"paired":paired,"future_rows_loaded":0},indent=2)+"\n",encoding="utf-8")
    print("Q_CORRECTION_INTERACTION_AUDIT_DONE");print((ROOT/"REPORT.txt").read_text())
if __name__=="__main__":main()
