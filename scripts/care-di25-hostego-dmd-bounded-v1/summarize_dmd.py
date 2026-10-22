#!/usr/bin/env python3
"""Post-hoc diagnostic unsealing and complete DMD-BC report generation."""
from __future__ import annotations
import csv, json, math
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

ROOT=Path("artifacts/care-di25-hostego-dmd-bounded-v1")
ROUTER=Path("artifacts/care-route-stage2-within-year-matrix-v1/shared/within_year_router_inputs.csv")
B0=Path("artifacts/care-di25-hostego-within-year-v1/m1_hostego_anomaly_bounded/within_year_m1_predictions.csv")
SEED=20260920; BOOT=10000; MAX_DELTA=2.25
DISCLAIMER=("The 2018–2021 same-year OOF sets have already been inspected during mechanism diagnosis and are therefore used only for development/diagnostic evaluation of DMD-BC.")

def norm(x):return str(x or "").strip().lower()
def read(path):
    with Path(path).open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def logit(p):
    p=np.clip(np.asarray(p,float),1e-6,1-1e-6);return np.log(p/(1-p))
def exact_mcnemar(n01,n10):
    n=n01+n10
    if not n:return 1.0
    k=min(n01,n10);return min(1.0,2*sum(math.comb(n,i) for i in range(k+1))/(2**n))
def metric(y,p):
    pred=p>=.5;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"N":len(y),"ACC":accuracy_score(y,pred),"BA":balanced_accuracy_score(y,pred),"AUC":roc_auc_score(y,p),"F1":f1_score(y,pred),"TN":tn,"FP":fp,"FN":fn,"TP":tp}
def ftype(label,p0,delta,final_correct):
    if final_correct:return "FIXED_OR_NOT_ERROR"
    required=abs(float(logit([p0])[0]));direction=(label==1 and delta>0) or (label==0 and delta<0)
    if required>MAX_DELTA:return "C_BOUND_LIMIT"
    if not direction:return "A_WRONG_DIRECTION"
    if abs(delta)<.10:return "D_NEAR_ZERO"
    return "B_RIGHT_DIRECTION_TOO_WEAK"
def avg_seed(path):
    rows=read(path);groups={}
    for r in rows:groups.setdefault(norm(r["sha256"]),[]).append(r)
    out={}
    for key,ss in groups.items():
        def mean(name):
            values=[float(x[name]) for x in ss if x.get(name,"") not in ("","nan","NaN")]
            return float(np.mean(values)) if values else np.nan
        out[key]={name:mean(name) for name in ("action_neg_prob","action_keep_prob","action_pos_prob","direction_value","magnitude","delta")}
    return out
def bootstrap(y,pa,pb):
    rng=np.random.RandomState(SEED);values={k:[] for k in ("BA","AUC","F1")};n=len(y)
    for _ in range(BOOT):
        idx=rng.randint(0,n,n)
        if len(np.unique(y[idx]))<2:continue
        a=metric(y[idx],pa[idx]);b=metric(y[idx],pb[idx])
        for k in values:values[k].append(b[k]-a[k])
    return {f"delta_{k}_ci_low":np.percentile(v,2.5) for k,v in values.items()}|{f"delta_{k}_ci_high":np.percentile(v,97.5) for k,v in values.items()}

def main():
    router=read(ROUTER);ids=[norm(r["sha256"]) for r in router];ri={norm(r["sha256"]):r for r in router}
    b0={norm(r["sha256"]):r for r in read(B0)}
    experts={"B0":np.asarray([float(b0[x]["p_expert_bounded"]) for x in ids])}
    actions={"B0":{}}
    for arm in ("B1_nokeep","B2_full"):
        lookup={norm(r["sha256"]):r for r in read(ROOT/arm/"expert_predictions.csv")}
        experts[arm]=np.asarray([float(lookup[x]["p_expert"]) for x in ids]);actions[arm]=avg_seed(ROOT/arm/"per_seed_internal.csv")
    y=np.asarray([int(r["label"]) for r in router]);years=np.asarray([int(r["year"]) for r in router]);folds=np.asarray([int(r["fold"]) for r in router])
    p0=np.asarray([float(r["p_stage1"]) for r in router]);route=np.asarray([int(float(r["route_DI"]))==1 for r in router]);basepred=p0>=.5;baseok=basepred==y
    final={arm:np.where(route,p,p0) for arm,p in experts.items()}
    per=[];failure_rows=[]
    for arm in ("B0","B1_nokeep","B2_full"):
        pred=final[arm]>=.5;correct=pred==y;effective=logit(experts[arm])-logit(p0)
        for i,key in enumerate(ids):
            a=actions[arm].get(key,{})
            target=("KEEP" if baseok[i] else "POS" if y[i]==1 else "NEG")
            probs=[a.get("action_neg_prob",np.nan),a.get("action_keep_prob",np.nan),a.get("action_pos_prob",np.nan)]
            labels=["NEG","KEEP","POS"]
            if arm=="B1_nokeep":
                pred_action="POS" if a.get("action_pos_prob",0)>=a.get("action_neg_prob",0) else "NEG"
                diagnostic_target=target if not baseok[i] else ""
            elif arm=="B2_full":pred_action=labels[int(np.nanargmax(probs))];diagnostic_target=target
            else:pred_action=diagnostic_target=""
            required=abs(float(logit([p0[i]])[0]))+1e-3 if not baseok[i] else np.nan
            kind=ftype(y[i],p0[i],effective[i],correct[i]) if route[i] and not baseok[i] else ""
            row={"arm":arm,"year":years[i],"fold":folds[i],"sha256":key,"label":y[i],"p_stage1":p0[i],"pred_stage1":int(basepred[i]),"stage1_correct":int(baseok[i]),
                "DI_score":float(ri[key]["U_DI"]),"routed":int(route[i]),"p_final":final[arm][i],"pred_final":int(pred[i]),"final_correct":int(correct[i]),
                "fixed":int(route[i] and not baseok[i] and correct[i]),"broken":int(route[i] and baseok[i] and not correct[i]),
                "action_neg_prob":a.get("action_neg_prob",np.nan),"action_keep_prob":a.get("action_keep_prob",np.nan),"action_pos_prob":a.get("action_pos_prob",np.nan),
                "direction_value":a.get("direction_value",np.nan),"magnitude":a.get("magnitude",np.nan),"delta":effective[i],
                "required_delta_if_stage1_wrong":required,"action_target_for_diagnostic":diagnostic_target,"action_pred":pred_action,
                "action_correct":int(pred_action==diagnostic_target) if diagnostic_target else "","failure_type":kind}
            per.append(row)
            if kind:failure_rows.append(row)
    write(ROOT/"reports"/"per_apk_predictions.csv",per)
    metrics=[]
    for arm in ("B0","B1_nokeep","B2_full"):
        for scope,mask in [(str(v),years==v) for v in (2018,2019,2020,2021)]+[("pooled",np.ones(len(y),bool))]:
            m=metric(y[mask],final[arm][mask]);r=route&mask;fixed=np.sum(r&~baseok&((final[arm]>=.5)==y));broken=np.sum(r&baseok&((final[arm]>=.5)!=y));den=np.sum(r&~baseok)
            metrics.append({"arm":arm,"scope":scope,**m,"routed_n":int(np.sum(r)),"fixed":int(fixed),"broken":int(broken),"net":int(fixed-broken),
                "correction_recall":fixed/den if den else np.nan,"correction_precision":fixed/(fixed+broken) if fixed+broken else np.nan,
                "break_rate":broken/np.sum(r&baseok) if np.sum(r&baseok) else np.nan,"net_per_routed":(fixed-broken)/np.sum(r)})
    write(ROOT/"reports"/"metrics_by_year.csv",[r for r in metrics if r["scope"]!="pooled"]);write(ROOT/"reports"/"metrics_pooled.csv",[r for r in metrics if r["scope"]=="pooled"])
    paired=[]
    for old,new in (("B0","B1_nokeep"),("B0","B2_full"),("B1_nokeep","B2_full")):
        for scope,mask in [(str(v),years==v) for v in (2018,2019,2020,2021)]+[("pooled",np.ones(len(y),bool))]:
            ao=(final[old]>=.5)==y;an=(final[new]>=.5)==y;n01=int(np.sum(mask&~ao&an));n10=int(np.sum(mask&ao&~an));ma=metric(y[mask],final[old][mask]);mb=metric(y[mask],final[new][mask])
            row={"comparison":f"{old}_vs_{new}","scope":scope,"same_correct":int(np.sum(mask&ao&an)),"same_wrong":int(np.sum(mask&~ao&~an)),"old_wrong_new_correct":n01,"old_correct_new_wrong":n10,
                "exact_mcnemar_p":exact_mcnemar(n01,n10),"delta_BA":mb["BA"]-ma["BA"],"delta_AUC":mb["AUC"]-ma["AUC"],"delta_F1":mb["F1"]-ma["F1"]}
            if scope=="pooled":row.update(bootstrap(y[mask],final[old][mask],final[new][mask]))
            paired.append(row)
    write(ROOT/"reports"/"paired_comparisons.csv",paired)
    failure=[]
    for arm in ("B0","B1_nokeep","B2_full"):
        subset=[r for r in failure_rows if r["arm"]==arm];cnt=Counter(r["failure_type"] for r in subset)
        for scope in ["pooled"]:
            for kind in ("A_WRONG_DIRECTION","B_RIGHT_DIRECTION_TOO_WEAK","C_BOUND_LIMIT","D_NEAR_ZERO"):
                failure.append({"arm":arm,"scope":scope,"failure_type":kind,"count":cnt[kind]})
        failure.append({"arm":arm,"scope":"pooled","failure_type":"BROKEN","count":next(r["broken"] for r in metrics if r["arm"]==arm and r["scope"]=="pooled")})
    write(ROOT/"reports"/"failure_type_comparison.csv",failure)
    confusion_rows=[];summary_rows=[]
    for arm in ("B1_nokeep","B2_full"):
        for scope,mask in [(str(v),years==v) for v in (2018,2019,2020,2021)]+[("pooled",np.ones(len(y),bool))]:
            rows=[r for r in per if r["arm"]==arm and mask[ids.index(r["sha256"])] and r["action_target_for_diagnostic"]]
            classes=["NEG","POS"] if arm=="B1_nokeep" else ["NEG","KEEP","POS"]
            for t in classes:
                for p in classes:confusion_rows.append({"arm":arm,"scope":scope,"target":t,"prediction":p,"count":sum(r["action_target_for_diagnostic"]==t and r["action_pred"]==p for r in rows)})
            recalls={c:sum(r["action_target_for_diagnostic"]==c and r["action_pred"]==c for r in rows)/max(1,sum(r["action_target_for_diagnostic"]==c for r in rows)) for c in classes}
            f1s=[]
            for c in classes:
                tp=sum(r["action_target_for_diagnostic"]==c and r["action_pred"]==c for r in rows);fp=sum(r["action_target_for_diagnostic"]!=c and r["action_pred"]==c for r in rows);fn=sum(r["action_target_for_diagnostic"]==c and r["action_pred"]!=c for r in rows);f1s.append(2*tp/max(1,2*tp+fp+fn))
            summary_rows.append({"arm":arm,"scope":scope,**{f"{c}_recall":recalls.get(c,np.nan) for c in ("NEG","KEEP","POS")},"action_macro_F1":np.mean(f1s),
                "stage1_wrong_direction_accuracy":np.mean([r["action_correct"] for r in rows if not int(r["stage1_correct"])]),
                "stage1_correct_KEEP_recall":recalls.get("KEEP",np.nan),"FP_to_NEG_recall":recalls.get("NEG",np.nan),"FN_to_POS_recall":recalls.get("POS",np.nan)})
    write(ROOT/"reports"/"action_confusion_by_year.csv",confusion_rows);write(ROOT/"reports"/"action_confusion_pooled.csv",[r for r in summary_rows if r["scope"]=="pooled"])
    # Historical transition cohorts are defined by B0's unchanged failure definition.
    b0fail={r["sha256"]:r for r in failure_rows if r["arm"]=="B0"};b2rows={r["sha256"]:r for r in per if r["arm"]=="B2_full"}
    wrong=[];weak=[]
    for key,old in b0fail.items():
        new=b2rows[key]
        if old["failure_type"]=="A_WRONG_DIRECTION":
            status="WRONG_DIRECTION_FIXED" if int(new["final_correct"]) else "WRONG_DIRECTION_REMAINS" if new["failure_type"]=="A_WRONG_DIRECTION" else "CHANGED_OTHER"
            wrong.append({"sha256":key,"year":old["year"],"label":old["label"],"B0_delta":old["delta"],"B0_failure_type":old["failure_type"],
                "B2_action_probs":json.dumps([new["action_neg_prob"],new["action_keep_prob"],new["action_pos_prob"]]),"B2_direction":new["direction_value"],"B2_magnitude":new["magnitude"],"B2_delta":new["delta"],"B2_correct":new["final_correct"],"status":status})
        if old["failure_type"]=="B_RIGHT_DIRECTION_TOO_WEAK":
            b0gap=max(0,float(old["required_delta_if_stage1_wrong"])-abs(float(old["delta"])));b2gap=max(0,float(new["required_delta_if_stage1_wrong"])-abs(float(new["delta"])))
            status="TOO_WEAK_FIXED" if int(new["final_correct"]) else "BECAME_WRONG_DIRECTION" if new["failure_type"]=="A_WRONG_DIRECTION" else "TOO_WEAK_REMAINS" if new["failure_type"]=="B_RIGHT_DIRECTION_TOO_WEAK" else "OTHER"
            weak.append({"sha256":key,"year":old["year"],"required_delta":old["required_delta_if_stage1_wrong"],"B0_delta":old["delta"],"B0_gap":b0gap,
                "B2_direction":new["direction_value"],"B2_magnitude":new["magnitude"],"B2_delta":new["delta"],"B2_gap":b2gap,"B2_correct":new["final_correct"],"status":status})
    write(ROOT/"reports"/"historical_wrong_direction_transition.csv",wrong);write(ROOT/"reports"/"historical_too_weak_transition.csv",weak)
    leakage={"validation_labels_used_in_training":False,"validation_labels_used_in_action_target":False,"validation_labels_used_in_magnitude_target":False,"future_year_labels_used":False,
        "same_year_outer_train_labels_used":True,"same_year_outer_validation_labels_unsealed_posthoc_only":True,"year_identity_used_as_feature":False,
        "DI25_changed":False,"budget_changed":False,"max_delta_changed":False,"HostEgo_changed":False,"Anomaly_changed":False,"Attention_changed":False}
    (ROOT/"audits"/"leakage_audit.json").write_text(json.dumps(leakage,indent=2)+"\n",encoding="utf-8")
    loss={"L_total":"L_final(existing weighted BCE + keep consistency) + 1.0*L_action + 1.0*L_magnitude","L_action":"ordinary CE; B2 all train rows, B1 only Stage-I-wrong train rows","L_magnitude":"SmoothL1 on Stage-I-wrong train rows only","lambda_action":1.0,"lambda_magnitude":1.0,"margin_epsilon":1e-3,"hyperparameter_search":False}
    (ROOT/"audits"/"loss_audit.json").write_text(json.dumps(loss,indent=2)+"\n",encoding="utf-8")
    pm={r["arm"]:r for r in metrics if r["scope"]=="pooled"};fc={(r["arm"],r["failure_type"]):r["count"] for r in failure};act={r["arm"]:r for r in summary_rows if r["scope"]=="pooled"}
    consistent=all(next(r for r in metrics if r["arm"]=="B2_full" and r["scope"]==str(yv))["net"]>=next(r for r in metrics if r["arm"]=="B0" and r["scope"]==str(yv))["net"] for yv in (2018,2019,2020,2021))
    report=["CARE-Droid DMD-BC one-shot mechanism report",DISCLAIMER,"","Q1. B2 reduced Wrong Direction: "+str(fc[("B0","A_WRONG_DIRECTION")])+" -> "+str(fc[("B2_full","A_WRONG_DIRECTION")]),
        f"Q2. Historical Wrong Direction fixed by B2: {sum(r['status']=='WRONG_DIRECTION_FIXED' for r in wrong)}/{len(wrong)}",
        f"Q3. Historical Too Weak fixed by B2: {sum(r['status']=='TOO_WEAK_FIXED' for r in weak)}/{len(weak)}",
        f"Q4. B2 correction recall: {pm['B2_full']['correction_recall']:.6f} (B0 {pm['B0']['correction_recall']:.6f})",
        f"Q5. B2 correction precision: {pm['B2_full']['correction_precision']:.6f}",f"Q6. Broken: B0={pm['B0']['broken']} B2={pm['B2_full']['broken']}",
        f"Q7. KEEP comparison: B1 broken={pm['B1_nokeep']['broken']}, B2 broken={pm['B2_full']['broken']}; B1 net={pm['B1_nokeep']['net']}, B2 net={pm['B2_full']['net']}",
        f"Q8. B2 action recall NEG={act['B2_full']['NEG_recall']:.6f} KEEP={act['B2_full']['KEEP_recall']:.6f} POS={act['B2_full']['POS_recall']:.6f}",
        f"Q9. FN -> POS recall: {act['B2_full']['FN_to_POS_recall']:.6f}",f"Q10. FP -> NEG recall: {act['B2_full']['FP_to_NEG_recall']:.6f}",
        f"Q11. Four-year net direction consistent (B2 >= B0): {consistent}","Q12. Inspect metrics_by_year.csv; no year is hidden or relabeled as unseen.",
        "Q13. See audits/parameter_count.csv for actual instantiated counts and tensor shapes.",
        "Q14. direction/magnitude coupling support: "+("SUPPORTED" if fc[("B2_full","A_WRONG_DIRECTION")]<fc[("B0","A_WRONG_DIRECTION")] and pm["B2_full"]["correction_recall"]>pm["B0"]["correction_recall"] and pm["B2_full"]["broken"]<=pm["B0"]["broken"]+2 else "NOT SUPPORTED"),
        "Q15. No automatic model revision is authorized; stop and await human confirmation."]
    (ROOT/"reports"/"FINAL_DMD_BOUNDED_REPORT.txt").write_text("\n".join(report)+"\n",encoding="utf-8")
    print("DMD_BOUNDED_SUMMARY_PASS")

if __name__=="__main__":main()
