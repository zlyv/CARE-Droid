#!/usr/bin/env python3
"""Summarize the predeclared same-year OOF Host-complement experiment."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit_common import (
    ARM_NAMES, EXPECTED_M0_SHA256, LOCKED_M0, ROOT, ROUTER, binary_metrics,
    exact_mcnemar, hash_manifest, normal, protected_paths, read_csv,
    router_contract, sha256_file, write_csv, write_json,
)


YEARS=(2018,2019,2020,2021)
METHODS=("Global24","M0_LOCKED","M_A_COMPOSITION","M_B_POSITION","M_AB_COMPOSITION_POSITION")
COMPARISONS=(
    ("M0_LOCKED","M_A_COMPOSITION"),("M0_LOCKED","M_B_POSITION"),("M0_LOCKED","M_AB_COMPOSITION_POSITION"),
    ("M_A_COMPOSITION","M_B_POSITION"),("M_A_COMPOSITION","M_AB_COMPOSITION_POSITION"),("M_B_POSITION","M_AB_COMPOSITION_POSITION"),
)


def indexed(path:Path)->dict[str,dict]:
    rows=read_csv(path); result={normal(r["sha256"]):r for r in rows}
    if len(rows)!=len(result): raise RuntimeError("duplicate SHA: "+str(path))
    return result


def correction(y,p,base,route):
    s=(base>=.5)==y;n=(p>=.5)==y
    fixed=int(np.sum(route & ~s & n));broken=int(np.sum(route & s & ~n));changed=int(np.sum(route & ((base>=.5)!=(p>=.5))))
    stage1_errors=int(np.sum(route & ~s));stage1_correct=int(np.sum(route & s))
    return {"Fixed":fixed,"Broken":broken,"Net":fixed-broken,"Changed":changed,
            "CorrectionPrecision":fixed/max(fixed+broken,1),"CorrectionRecall":fixed/max(stage1_errors,1),"BreakRate":broken/max(stage1_correct,1),
            "routed_stage1_errors":stage1_errors,"routed_stage1_correct":stage1_correct}


def fast_metric(y,p):
    from sklearn.metrics import balanced_accuracy_score,f1_score,roc_auc_score
    q=p>=.5
    return np.array([balanced_accuracy_score(y,q),roc_auc_score(y,p),f1_score(y,q,zero_division=0)],float)


def bootstrap(y,years,probs,n=10000,seed=20260920):
    rng=np.random.default_rng(seed);strata=[np.flatnonzero((years==yr)&(y==label)) for yr in YEARS for label in (0,1)]
    names=list(probs);values={k:np.empty((n,3),float) for k in names}
    for b in range(n):
        ix=np.concatenate([rng.choice(s,len(s),replace=True) for s in strata])
        for name in names:values[name][b]=fast_metric(y[ix],probs[name][ix])
    rows=[]
    for old,new in COMPARISONS:
        d=values[new]-values[old]
        row={"baseline":old,"candidate":new,"bootstrap_iterations":n,"bootstrap_seed":seed}
        for j,m in enumerate(("BA","AUC","F1")):
            point=fast_metric(y,probs[new])[j]-fast_metric(y,probs[old])[j]
            row[f"Delta_{m}"]=float(point);row[f"CI95_low_{m}"]=float(np.quantile(d[:,j],.025));row[f"CI95_high_{m}"]=float(np.quantile(d[:,j],.975))
        rows.append(row)
    return rows


def historical_results(ids,y,prob):
    candidates=[
        Path("artifacts/care-di25-hostego-complementarity-audit-v1/reports/wrong_direction_graph_missed_samples.csv"),
        Path("artifacts/care-di25-host-complement-candidate-audit-v1/reports/wrong_direction_graph_missed_samples.csv"),
    ]
    wd=next((p for p in candidates if p.is_file()),None)
    failures=Path("artifacts/care-di25-hostego-v1/stage2_mechanism_diagnosis/routed_unfixed_failure_types.csv")
    if wd is None or not failures.is_file():
        raise RuntimeError("BLOCKED_BY_CONTRACT: locked historical failure mapping is unavailable")
    index={s:i for i,s in enumerate(ids)}
    failure_rows=read_csv(failures)
    if sum(r.get("failure_type")=="A_WRONG_DIRECTION" for r in failure_rows)!=27 or sum(r.get("failure_type")=="B_RIGHT_DIRECTION_TOO_WEAK" for r in failure_rows)!=29:
        raise RuntimeError("BLOCKED_BY_CONTRACT: historical 27/29 cohorts changed")
    detailed=[]
    for row in failure_rows:
        sha=normal(row["sha256"]);i=index[sha]
        item={"sha256":sha,"year":int(row["year"]),"fold":int(row["fold"]),"label":int(y[i]),"failure_type":row["failure_type"]}
        for name in ("M0_LOCKED","M_A_COMPOSITION","M_B_POSITION","M_AB_COMPOSITION_POSITION"):
            item[name+"_correct"]=int((prob[name][i]>=.5)==y[i])
        detailed.append(item)
    wd_rows=read_csv(wd)
    if len(wd_rows)!=20: raise RuntimeError("BLOCKED_BY_CONTRACT: graph-missed WD cohort is not 20")
    gm=[];existing=[]
    for row in wd_rows:
        sha=normal(row["sha256"]);i=index[sha]
        item={"sha256":sha,"year":int(row["year"]),"fold":int(row["fold"]),"label":int(y[i])}
        for name in ("M0_LOCKED","M_A_COMPOSITION","M_B_POSITION","M_AB_COMPOSITION_POSITION"):
            item[name+"_correct"]=int((prob[name][i]>=.5)==y[i])
        gm.append(item)
        crit=int(float(row.get("critical_positive",float(row.get("critical_hostego_score",0))>=.5)))
        att=int(float(row.get("attended_positive",float(row.get("attended_hostego_score",0))>=.5)))
        if crit==0 and att==0:existing.append(item)
    if len(existing)!=3: raise RuntimeError("BLOCKED_BY_CONTRACT: ExistingHostMiss WD cohort is not 3")
    return detailed,gm,existing


def decision(m0,variant,boot,mcnemar,annual):
    d={k:variant[k]-m0[k] for k in ("BA","AUC","F1")};net=variant["Net"]-m0["Net"]
    ba_low=float(boot["CI95_low_BA"]);favorable=mcnemar["old_wrong_new_correct"]>=mcnemar["old_correct_new_wrong"]
    consistent=sum(r["DeltaNet_vs_M0"]>=0 for r in annual)>=3
    if d["BA"]>0 and d["F1"]>=0 and net>=0 and favorable and ba_low>0 and consistent:return "SUPPORTED"
    if d["BA"]<=0 and d["F1"]<=0 and net<0:return "NOT_SUPPORTED"
    return "MIXED"


def main():
    if sha256_file(LOCKED_M0)!=EXPECTED_M0_SHA256:raise RuntimeError("BLOCKED_BY_CONTRACT: M0 SHA changed")
    routes=read_csv(ROUTER);rc=router_contract(routes)
    if not rc["passed"]:raise RuntimeError("BLOCKED_BY_CONTRACT: router changed")
    ids=[normal(r["sha256"]) for r in routes];y=np.asarray([int(r["label"]) for r in routes]);years=np.asarray([int(r["year"]) for r in routes]);base=np.asarray([float(r["p_stage1"]) for r in routes]);route=np.asarray([int(r["route_DI"])==1 for r in routes])
    m0=indexed(LOCKED_M0);experts={"M0_LOCKED":np.asarray([float(m0[s]["p_expert_bounded"]) for s in ids])}
    paths={"M_A_COMPOSITION":ROOT/"predictions/M_A_expert_predictions.csv","M_B_POSITION":ROOT/"predictions/M_B_expert_predictions.csv","M_AB_COMPOSITION_POSITION":ROOT/"predictions/M_AB_expert_predictions.csv"}
    arm_rows={}
    for name,path in paths.items():
        z=indexed(path);arm_rows[name]=z
        if set(z)!=set(ids):raise RuntimeError("arm cohort mismatch: "+name)
        experts[name]=np.asarray([float(z[s]["p_expert"]) for s in ids])
    final={"Global24":base,"M0_LOCKED":np.where(route,experts["M0_LOCKED"],base)}
    final.update({k:np.where(route,v,base) for k,v in experts.items() if k!="M0_LOCKED"})
    for name,p in final.items():
        if name!="Global24" and not np.array_equal(p[~route],base[~route]):raise RuntimeError("unrouted probabilities changed")
    metrics=[];corrections=[]
    for scope,mask in [(str(yv),years==yv) for yv in YEARS]+[("Pooled",np.ones(len(y),bool))]:
        for name,p in final.items():
            metric={"scope":scope,"method":name,"n":int(mask.sum()),**binary_metrics(y[mask],p[mask])};metrics.append(metric)
            method_route=np.zeros(int(mask.sum()),bool) if name=="Global24" else route[mask]
            corrections.append({"scope":scope,"method":name,"routed":int(method_route.sum()),**correction(y[mask],p[mask],base[mask],method_route)})
    pooled={r["method"]:{**r,**next(x for x in corrections if x["scope"]=="Pooled" and x["method"]==r["method"])} for r in metrics if r["scope"]=="Pooled"}
    mcn=[]
    for old,new in COMPARISONS:
        oc=(final[old]>=.5)==y;nc=(final[new]>=.5)==y;mcn.append({"scope":"Pooled","baseline":old,"candidate":new,**exact_mcnemar(oc,nc)})
    boots=bootstrap(y,years,{k:final[k] for k in ("M0_LOCKED","M_A_COMPOSITION","M_B_POSITION","M_AB_COMPOSITION_POSITION")})
    detailed,gm,existing=historical_results(ids,y,final)
    pred_rows=[]
    for i,s in enumerate(ids):
        row={"sha256":s,"year":int(years[i]),"label":int(y[i]),"fold":int(routes[i]["fold"]),"p_stage1":base[i],"route_DI":int(route[i])}
        for name,p in final.items():row["p_final_"+name]=p[i]
        pred_rows.append(row)
    reports=ROOT/"reports";pred_dir=ROOT/"predictions"
    write_csv(pred_dir/"end_to_end_predictions.csv",pred_rows);write_csv(reports/"end_to_end_metrics.csv",metrics);write_csv(reports/"correction_metrics.csv",corrections);write_csv(reports/"paired_mcnemar.csv",mcn);write_csv(reports/"paired_bootstrap.csv",boots);write_csv(reports/"historical_failure_cohort.csv",detailed);write_csv(reports/"graph_missed_wd_results.csv",gm);write_csv(reports/"existing_host_miss_results.csv",existing)
    # Actual instantiated parameter counts are inherited from each completed fold audit.
    pc=[{"method":"M0_LOCKED","parameter_count":34613,"extra_vs_M0":0,"relative_increase":0.0}]
    for name,z in arm_rows.items():
        counts={int(float(r["parameter_count"])) for r in z.values()};
        if len(counts)!=1:raise RuntimeError("parameter count mismatch: "+name)
        count=next(iter(counts));pc.append({"method":name,"parameter_count":count,"extra_vs_M0":count-34613,"relative_increase":count/34613-1})
    write_csv(reports/"parameter_count.csv",pc)
    comp=[];position=[];id_to_index={s:i for i,s in enumerate(ids)}
    for name,z in arm_rows.items():
        if name in ("M_A_COMPOSITION","M_AB_COMPOSITION_POSITION"):
            for s,r in z.items():
                i=id_to_index[s];comp.append({"method":name,"sha256":s,"year":r["year"],"label":r["label"],"routed":int(routes[i]["route_DI"]),"stage1_correct":int((float(routes[i]["p_stage1"])>=.5)==int(r["label"])),"mean_alpha":r["mean_alpha"],"median_alpha":r["median_alpha"],"max_alpha":r["max_alpha"],"multi_role_host_fraction":r["multi_role_host_fraction"],"has_nonzero_composition":int(int(r["multi_role_host_count"])>0)})
        if name in ("M_B_POSITION","M_AB_COMPOSITION_POSITION"):
            for s,r in z.items():
                i=id_to_index[s];position.append({"method":name,"sha256":s,"year":r["year"],"label":r["label"],"routed":int(routes[i]["route_DI"]),"mean_beta":r["mean_beta"],"median_beta":r["median_beta"],"max_beta":r["max_beta"],"source_like_hosts":r["source_like_host_count"],"sink_like_hosts":r["sink_like_host_count"],"bridge_like_hosts":r["bridge_like_host_count"],"isolated_like_hosts":r["isolated_like_host_count"]})
    write_csv(reports/"composition_usage.csv",comp);write_csv(reports/"position_usage.csv",position)
    byboot={(r["baseline"],r["candidate"]):r for r in boots};bymc={(r["baseline"],r["candidate"]):r for r in mcn}
    results={}
    for short,name in (("A","M_A_COMPOSITION"),("B","M_B_POSITION"),("AB","M_AB_COMPOSITION_POSITION")):
        annual=[]
        for yr in map(str,YEARS):
            vr=next(r for r in corrections if r["method"]==name and r["scope"]==yr)
            mr=next(r for r in corrections if r["method"]=="M0_LOCKED" and r["scope"]==yr)
            annual.append({"year":yr,"DeltaNet_vs_M0":vr["Net"]-mr["Net"]})
        results[short]=decision(pooled["M0_LOCKED"],pooled[name],byboot[("M0_LOCKED",name)],bymc[("M0_LOCKED",name)],annual)
    supported=[k for k,v in results.items() if v=="SUPPORTED"]
    primary=("NO_VARIANT_SUPPORTED" if not supported and all(v=="NOT_SUPPORTED" for v in results.values()) else ("M_"+supported[0] if len(supported)==1 else "NO_UNIQUE_WINNER"))
    gm_counts={name:sum(int(r[name+"_correct"]) for r in gm) for name in ("M0_LOCKED","M_A_COMPOSITION","M_B_POSITION","M_AB_COMPOSITION_POSITION")}
    ex_fixed={name:sum(int(r[name+"_correct"]) and not int(r["M0_LOCKED_correct"]) for r in existing) for name in ("M_A_COMPOSITION","M_B_POSITION","M_AB_COMPOSITION_POSITION")}
    lines=["CARE-Droid Host Complement Neural Integration v1","Same-year OOF development/diagnostic only; not future, unseen, zero-day, or independent temporal evidence.","","Pooled:"]
    for name in METHODS:
        r=pooled[name];lines.append(f"{name}: BA={r['BA']:.6f} AUC={r['AUC']:.6f} F1={r['F1']:.6f} Fixed={r['Fixed']} Broken={r['Broken']} Net={r['Net']}")
    lines += ["",f"Graph-missed WrongDirection: N={len(gm)} "+" ".join(f"{k} correct={v}" for k,v in gm_counts.items()),f"ExistingHostMiss WrongDirection: N={len(existing)} "+" ".join(f"{k} fixed={v}" for k,v in ex_fixed.items()),"",f"A_RESULT={results['A']}",f"B_RESULT={results['B']}",f"AB_RESULT={results['AB']}",f"PRIMARY_VARIANT={primary}","Capacity increase is a possible confound, especially for AB; no capacity-matched arm was run.","All original SHA256 unchanged.","HOST_COMPLEMENT_NEURAL_EXPERIMENT_DONE"]
    (reports/"HOST_COMPLEMENT_NEURAL_REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (reports/"HOST_COMPLEMENT_NEURAL_DECISION.txt").write_text("\n".join(lines[-7:])+"\n",encoding="utf-8")
    after=hash_manifest(protected_paths());before=json.loads((ROOT/"preflight/original_sha256_before.json").read_text(encoding="utf-8"));write_json(ROOT/"audits/no_original_modification.json",{"passed":before==after,"before":before,"after":after})
    if before!=after:raise RuntimeError("BLOCKED_BY_CONTRACT: protected original SHA changed")
    print("\n".join(lines))


if __name__=="__main__":main()
