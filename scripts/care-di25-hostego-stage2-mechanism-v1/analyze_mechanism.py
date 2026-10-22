#!/usr/bin/env python3
"""Hard-gated mechanism analysis from the frozen M1 instrumentation rerun."""
from __future__ import annotations

import csv, json, math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score, confusion_matrix

ROOT=Path("artifacts/care-di25-hostego-v1/stage2_mechanism_diagnosis")
RUN=ROOT/"instrumented_rerun"
ROUTER=Path("artifacts/care-route-stage2-within-year-matrix-v1/shared/within_year_router_inputs.csv")
MAX_DELTA=2.25
DISCLAIMER=("Independent within-year five-fold OOF mechanism diagnosis. It is not temporal-generalization, "
            "future-year, zero-day, or unseen-future evidence.")


def read_csv(path):
    with Path(path).open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def write_csv(path,rows):
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with Path(path).open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def normal(x):return str(x or "").strip().lower()
def logit(p):
    p=np.clip(np.asarray(p,dtype=np.float64),1e-5,1-1e-5)
    return np.log(p/(1-p)).astype(np.float32)
def cohort(s1,routed,final):
    return ("A" if s1 else "B") if not routed else (("C" if final else "D") if not s1 else ("E" if final else "F"))
def stats(values):
    x=np.asarray(values,float);x=x[np.isfinite(x)]
    return {"n":len(x),"mean":float(np.mean(x)) if len(x) else np.nan,"median":float(np.median(x)) if len(x) else np.nan,
            "std":float(np.std(x)) if len(x) else np.nan,"iqr":float(np.percentile(x,75)-np.percentile(x,25)) if len(x) else np.nan}
def metric(y,p):
    pred=np.asarray(p)>=.5;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"BA":float(balanced_accuracy_score(y,pred)),"AUC":float(roc_auc_score(y,p)),"F1":float(f1_score(y,pred)),
            "TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp)}


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    apps=read_csv(RUN/"apk_internal.csv"); hosts=read_csv(RUN/"host_internal.csv"); router=read_csv(ROUTER)
    if len(apps)!=1989 or len({normal(r["sha256"]) for r in apps})!=1989:
        raise RuntimeError("BLOCKED_BY_REPRODUCTION: instrumented cohort is not exact 1989")
    ai={normal(r["sha256"]):r for r in apps}; ri={normal(r["sha256"]):r for r in router}
    if set(ai)!=set(ri):raise RuntimeError("BLOCKED_BY_REPRODUCTION: instrumented/router cohorts differ")
    ordered=[ai[normal(r["sha256"])] for r in router]
    y=np.asarray([int(float(r["label"])) for r in router]);p0=np.asarray([float(r["p_stage1"]) for r in router])
    route=np.asarray([int(float(r["route_DI"]))==1 for r in router]);old=np.asarray([float(r["locked_M1_prob"]) for r in ordered])
    new=np.asarray([float(r["rerun_M1_prob"]) for r in ordered]);old_final=np.where(route,old,p0);new_final=np.where(route,new,p0)
    diff=np.abs(new-old); pred_diff=(new>=.5)!=(old>=.5)
    identity=[]
    for i,r in enumerate(router):identity.append({"sha256":normal(r["sha256"]),"year":int(r["year"]),"fold":int(r["fold"]),
        "locked_probability":old[i],"rerun_probability":new[i],"abs_probability_diff":diff[i],
        "locked_pred":int(old[i]>=.5),"rerun_pred":int(new[i]>=.5),"prediction_diff":int(pred_diff[i])})
    write_csv(ROOT/"prediction_identity_audit.csv",identity)
    oldm,newm=metric(y,old_final),metric(y,new_final);baseok=(p0>=.5)==y
    def fb(p):
        ok=(p>=.5)==y;return int(np.sum(route&~baseok&ok)),int(np.sum(route&baseok&~ok))
    oldfb,newfb=fb(old_final),fb(new_final)
    exact_reproduction=bool(float(np.max(diff))<=1e-8 and int(np.sum(pred_diff))==0 and oldm==newm and oldfb==newfb)
    discrete_equal=bool(oldm["BA"]==newm["BA"] and oldm["F1"]==newm["F1"] and
                        all(oldm[k]==newm[k] for k in ("TN","FP","FN","TP")) and oldfb==newfb)
    numeric_equivalent=bool(int(np.sum(pred_diff))==0 and discrete_equal and float(np.max(diff))<=0.002)
    replay_status="EXACT_STATE_REPRODUCTION" if exact_reproduction else "NUMERICALLY_EQUIVALENT_REPLAY" if numeric_equivalent else "BLOCKED_BY_REPRODUCTION"
    summary={"replay_status":replay_status,"max_abs_prob_diff":float(np.max(diff)),"mean_abs_prob_diff":float(np.mean(diff)),
             "median_abs_prob_diff":float(np.median(diff)),"p95_abs_prob_diff":float(np.percentile(diff,95)),
             "p99_abs_prob_diff":float(np.percentile(diff,99)),"over_1e8_count":int(np.sum(diff>1e-8)),
             "over_1e4_count":int(np.sum(diff>1e-4)),"over_1e3_count":int(np.sum(diff>1e-3)),
             "prediction_difference_count":int(np.sum(pred_diff)),
             "locked_metrics":oldm,"rerun_metrics":newm,"locked_fixed_broken_net":[oldfb[0],oldfb[1],oldfb[0]-oldfb[1]],
             "rerun_fixed_broken_net":[newfb[0],newfb[1],newfb[0]-newfb[1]]}
    (ROOT/"prediction_identity_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    if not numeric_equivalent:
        raise RuntimeError("BLOCKED_BY_REPRODUCTION: "+json.dumps(summary))

    # Internal consistency is checked both at seed level and at ensemble-effective level.
    max_seed=max(float(r["seed_tanh_max_abs_diff"]) for r in ordered)
    bounded=np.asarray([float(r["bounded_delta"]) for r in ordered]);raw=np.asarray([float(r["raw_delta"]) for r in ordered])
    d1=np.max(np.abs(bounded-(logit(new)-logit(p0))))
    d2=np.max(np.abs(bounded-MAX_DELTA*np.tanh(raw)))
    if max(max_seed,d1,d2)>1e-6:raise RuntimeError(f"BLOCKED_BY_INTERNAL_CONSISTENCY: seed={max_seed} logit={d1} tanh={d2}")

    # Cohorts and APK-level summaries.
    rows=[]
    for i,(rr,a) in enumerate(zip(router,ordered)):
        # Cohort membership is always anchored to the locked historical M1
        # prediction. Replay probabilities never redefine fixed/unfixed/broken.
        s1=bool(baseok[i]); final=bool((old_final[i]>=.5)==y[i]); c=cohort(s1,bool(route[i]),final)
        required=float(-logit([p0[i]])[0])
        direction_ok=int((y[i]==1 and bounded[i]>0) or (y[i]==0 and bounded[i]<0))
        row={**a,"DI_score":float(rr["U_DI"]),"cohort":c,"Global24_correct":int(s1),"M1_correct":int(final),
             "required_flip_delta":required,"required_flip_abs":abs(required),"correction_direction_correct":direction_ok,
             "near_bound":int(abs(bounded[i])>=.95*MAX_DELTA)}
        rows.append(row)
    write_csv(ROOT/"apk_stage2_internal.csv",rows)

    # Add cohort to host rows and publish the required host-level diagnostic.
    bysha={r["sha256"]:r for r in rows}; host_out=[]
    for h in hosts:
        a=bysha[normal(h["sha256"])]; host_out.append({**h,"cohort":a["cohort"],"p_stage1":a["p_stage1"],
            "M1_prob":a["rerun_M1_prob"],"bounded_delta":a["bounded_delta"]})
    write_csv(ROOT/"host_internal_diagnostics.csv",host_out)

    # Mutually exclusive D failure types in predeclared priority order.
    failure=[]
    for r in rows:
        if r["cohort"]!="D":continue
        req=float(r["required_flip_abs"]);delta=float(r["bounded_delta"]);direction=bool(int(r["correction_direction_correct"]))
        flags={"flag_bound_limited":int(req>MAX_DELTA),"flag_wrong_direction":int(not direction),
               "flag_near_zero":int(abs(delta)<.10),"flag_right_direction_too_weak":int(direction and abs(delta)>=.10 and req<=MAX_DELTA)}
        kind=("C_BOUND_LIMIT" if flags["flag_bound_limited"] else "A_WRONG_DIRECTION" if flags["flag_wrong_direction"]
              else "D_NEAR_ZERO" if flags["flag_near_zero"] else "B_RIGHT_DIRECTION_TOO_WEAK")
        failure.append({**r,**flags,"failure_type":kind,"final_error_type":"FN" if int(r["label"])==1 else "FP"})
    if len(failure)!=74:raise RuntimeError(f"BLOCKED_BY_CONTRACT: expected 74 D rows, found {len(failure)}")
    write_csv(ROOT/"routed_unfixed_failure_types.csv",failure)
    write_csv(ROOT/"routed_unfixed_FN_analysis.csv",[r for r in failure if r["final_error_type"]=="FN"])
    write_csv(ROOT/"routed_unfixed_FP_analysis.csv",[r for r in failure if r["final_error_type"]=="FP"])
    if sum(r["final_error_type"]=="FN" for r in failure)!=48 or sum(r["final_error_type"]=="FP" for r in failure)!=26:
        raise RuntimeError("BLOCKED_BY_CONTRACT: routed-unfixed FN/FP counts changed from 48/26")
    broken=[r for r in rows if r["cohort"]=="F"]
    if len(broken)!=11:raise RuntimeError(f"BLOCKED_BY_CONTRACT: expected 11 broken rows, found {len(broken)}")
    write_csv(ROOT/"broken_internal_analysis.csv",broken)
    ft=[]
    for scope,subset in [(str(yv),[r for r in failure if int(r["year"])==yv]) for yv in (2018,2019,2020,2021)]+[("pooled",failure)]:
        for err in ("ALL","FN","FP"):
            ss=subset if err=="ALL" else [r for r in subset if r["final_error_type"]==err]
            cnt=Counter(r["failure_type"] for r in ss)
            for kind in ("A_WRONG_DIRECTION","B_RIGHT_DIRECTION_TOO_WEAK","C_BOUND_LIMIT","D_NEAR_ZERO"):
                ft.append({"scope":scope,"error_type":err,"failure_type":kind,"count":cnt[kind],"denominator":len(ss),
                           "rate":cnt[kind]/len(ss) if ss else np.nan})
    write_csv(ROOT/"failure_type_by_year.csv",ft)

    features=["bounded_delta","raw_delta","mean_gamma","max_gamma","mean_anomaly","max_anomaly","attention_entropy",
              "max_attention","real_host_count","nontrivial_ego_count"]
    comp=[]
    for scope,ss in [(str(yv),[r for r in rows if int(r["year"])==yv]) for yv in (2018,2019,2020,2021)]+[("pooled",rows)]:
        for c in ("C","D","F","E"):
            sub=[r for r in ss if r["cohort"]==c]
            for f in features:
                values=[float(r[f]) for r in sub if r.get(f,"") not in ("",None)]
                comp.append({"scope":scope,"cohort":c,"feature":f,**stats(values)})
    write_csv(ROOT/"cohort_internal_comparison.csv",comp)

    yearcmp=[]
    for scope,ss in [("2019_2020",[r for r in rows if int(r["year"]) in (2019,2020)]),("2021",[r for r in rows if int(r["year"])==2021])]:
        for c in ("C","D","F","E"):
            sub=[r for r in ss if r["cohort"]==c]
            for f in features:yearcmp.append({"scope":scope,"cohort":c,"feature":f,**stats([float(r[f]) for r in sub if r.get(f,"") not in ("",None)])})
    write_csv(ROOT/"year_stage2_mechanism_comparison.csv",yearcmp)

    # Critical-role and critical-Ego-pair recurrence (presence is counted per APK).
    critical=[h for h in host_out if int(float(h["critical_host"]))==1]
    role_table=[]
    for role in sorted(set(h["center_role"] for h in critical)):
        sub=[h for h in critical if h["center_role"]==role];years=sorted(set(int(h["year"]) for h in sub))
        role_table.append({"critical_role":role,"apk_count":len({h["sha256"] for h in sub}),"years":json.dumps(years),
                           "year_count":len(years),"recurs_at_least_3_years":int(len(years)>=3),
                           "D_count":sum(bysha[h["sha256"]]["cohort"]=="D" for h in sub),"F_count":sum(bysha[h["sha256"]]["cohort"]=="F" for h in sub)})
    write_csv(ROOT/"critical_role_analysis.csv",role_table)
    pairs=[]
    host_lookup={(h["sha256"],int(h["host_index"])):h for h in host_out}
    for h in critical:
        neighbors=[]
        for direction,key in (("in","incoming_neighbor_roles"),("out","outgoing_neighbor_roles")):
            index_key="incoming_neighbor_indices" if direction=="in" else "outgoing_neighbor_indices"
            for index,role in zip(json.loads(h[index_key]),json.loads(h[key])):
                neighbor=host_lookup.get((h["sha256"],int(index)))
                weight=float(neighbor["attention_weight"]) if neighbor else -np.inf
                neighbors.append((weight,(h["center_role"],role,direction)))
        if neighbors:
            # One deterministic top neighbor per critical Host; ties use tuple order.
            top=max(neighbors,key=lambda value:(value[0],value[1]))
            pairs.append((top[1],h,top[0]))
    pair_table=[]
    for pair in sorted(set(p for p,_,_ in pairs)):
        sub=[h for p,h,_ in pairs if p==pair];years=sorted(set(int(h["year"]) for h in sub))
        pair_table.append({"center_role":pair[0],"neighbor_role":pair[1],"direction":pair[2],
            "apk_count":len({h["sha256"] for h in sub}),"years":json.dumps(years),"year_count":len(years),
            "recurs_at_least_3_years":int(len(years)>=3),"D_count":sum(bysha[h["sha256"]]["cohort"]=="D" for h in sub),
            "definition":"top neighbor by ensemble-mean attention weight among critical Host neighbors"})
    write_csv(ROOT/"critical_ego_pair_analysis.csv",pair_table)

    counts=Counter(r["failure_type"] for r in failure);dominant,count=counts.most_common(1)[0]
    mapping={"C_BOUND_LIMIT":"BOUND_LIMIT","A_WRONG_DIRECTION":"EVIDENCE_DIRECTION",
             "D_NEAR_ZERO":"EVIDENCE_STRENGTH","B_RIGHT_DIRECTION_TOO_WEAK":"EVIDENCE_STRENGTH"}
    primary=mapping[dominant];years=sorted(set(int(r["year"]) for r in failure if r["failure_type"]==dominant))
    dominant_rows=[r for r in failure if r["failure_type"]==dominant]
    dominant_fn=sum(r["final_error_type"]=="FN" for r in dominant_rows); dominant_fp=count-dominant_fn
    def cmean(group,feature):
        v=[float(r[feature]) for r in rows if r["cohort"]==group and r.get(feature,"") not in ("",None)]
        return float(np.mean(v)) if v else np.nan
    internal_diffs={f:cmean("D",f)-cmean("C",f) for f in ("mean_gamma","mean_anomaly","max_attention","attention_entropy")}
    qualifies=count>=math.ceil(.25*len(failure)) or len(years)>=3
    intervention=[]
    if qualifies:
        intervention.append({"target":primary,"evidence":f"{dominant}: {count}/74 ({count/74:.1%}), FN={dominant_fn}, FP={dominant_fp}, years={years}; D-minus-C internal means={internal_diffs}",
            "plan":"Run one predeclared ablation targeted at this mechanism while keeping DI25, 25% budget, folds, seeds and max_delta fixed.",
            "risk":"Mechanism association is diagnostic, not causal; confirm with paired OOF predictions."})
    md=["# Stage-II targeted intervention plan","",DISCLAIMER,"",f"Primary observed failure: **{primary}**.",""]
    for i,x in enumerate(intervention,1):md += [f"## {i}. {x['target']}","",f"Evidence: {x['evidence']}","",f"Plan: {x['plan']}","",f"Risk: {x['risk']}",""]
    if not intervention:md += ["No intervention passed the predeclared recurrence/prevalence gate.",""]
    (ROOT/"STAGE2_TARGETED_INTERVENTION_PLAN.md").write_text("\n".join(md),encoding="utf-8")
    report=["CARE-Droid frozen M1 Stage-II mechanism diagnosis",DISCLAIMER,"",
            "EXACT_STATE_REPRODUCTION="+("PASS" if exact_reproduction else "FAIL"),
            "NUMERICALLY_EQUIVALENT_REPLAY=PASS",
            f"replay_status={replay_status}",
            "cohort_anchor=locked historical M1 probabilities; replay internals are descriptive only",
            f"max_abs_prob_diff={summary['max_abs_prob_diff']:.12g}",f"mean_abs_prob_diff={summary['mean_abs_prob_diff']:.12g}",
            f"p95_abs_prob_diff={summary['p95_abs_prob_diff']:.12g}",f"p99_abs_prob_diff={summary['p99_abs_prob_diff']:.12g}",
            f"prediction_difference_count={summary['prediction_difference_count']}",
            f"internal_seed_tanh_max_abs_diff={max_seed:.12g}",f"internal_ensemble_logit_max_abs_diff={d1:.12g}",
            f"internal_ensemble_tanh_max_abs_diff={d2:.12g}","",f"routed_unfixed_D={len(failure)} (FN={sum(r['final_error_type']=='FN' for r in failure)}, FP={sum(r['final_error_type']=='FP' for r in failure)})",
            f"broken_F={len(broken)}",f"failure_counts={dict(counts)}",f"PRIMARY_STAGE2_FAILURE={primary}",
            f"primary_evidence={dominant} {count}/74 ({count/74:.1%}), FN={dominant_fn}, FP={dominant_fp}, observed_years={years}",
            f"D_minus_C_internal_mean_differences={internal_diffs}","",
            "Year contrast (failure types):"]
    for yv in (2019,2020,2021):
        yc=Counter(r["failure_type"] for r in failure if int(r["year"])==yv)
        report.append(f"  {yv}: {dict(yc)}")
    report += ["",
            "Interpretation boundary: internal values come from a numerically equivalent replay, not the unavailable historical checkpoint. Cohorts and outcomes come from locked predictions.",
            "The primary label is assigned by the predeclared correction-failure hierarchy. Internal associations do not establish causal necessity or exact per-APK historical state."]
    (ROOT/"FINAL_STAGE2_MECHANISM_REPORT.txt").write_text("\n".join(report)+"\n",encoding="utf-8")
    print("STAGE2_MECHANISM_DIAGNOSIS_PASS")


if __name__=="__main__":main()
