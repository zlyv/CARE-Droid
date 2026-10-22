#!/usr/bin/env python3
"""Post-hoc mapping of frozen OOF need-correction scores to locked M1 cohorts."""
from __future__ import annotations
import csv,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT=Path("artifacts/care-di25-hostego-v1/stage2_mechanism_diagnosis")
COMP=Path("artifacts/care-di25-hostego-dmd-bounded-v1/r4_component_audit")
OUT=Path("artifacts/care-di25-hostego-dmd-bounded-v1/gate_cohort_audit")
PRED=COMP/"component_probe_oof_predictions.csv"
MECH=ROOT/"apk_stage2_internal.csv"
FAIL=ROOT/"routed_unfixed_failure_types.csv"
REPS=("S0_stage1_logit","S1_global24_plus_logit","S2_stage2_tail88","S3_tail88_plus_logit","S4_full_R4")
RNG_SEED=20260920

def norm(x):return str(x or "").strip().lower()
def read(path):
    with Path(path).open("r",encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with path.open("w",encoding="utf-8",newline="") as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")

def stats(values):
    x=np.asarray(values,float)
    return {"mean":float(np.mean(x)),"median":float(np.median(x)),"std":float(np.std(x)),
            "q25":float(np.percentile(x,25)),"q75":float(np.percentile(x,75)),
            "threshold_0p5_positive_n":int(np.sum(x>=.5)),"threshold_0p5_positive_rate":float(np.mean(x>=.5))}

def bootstrap_difference(a,b,iterations=10000):
    """Mean score a-b; descriptive APK bootstrap, no threshold selection."""
    a=np.asarray(a,float);b=np.asarray(b,float);rng=np.random.default_rng(RNG_SEED);v=[]
    for _ in range(iterations):
        aa=a[rng.integers(0,len(a),len(a))];bb=b[rng.integers(0,len(b),len(b))];v.append(float(np.mean(aa)-np.mean(bb)))
    v=np.asarray(v)
    return {"mean_score_difference":float(np.mean(a)-np.mean(b)),"bootstrap_mean_difference":float(np.mean(v)),
            "ci_low":float(np.percentile(v,2.5)),"ci_high":float(np.percentile(v,97.5)),
            "iterations":iterations,"seed":RNG_SEED}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    predictions=[r for r in read(PRED) if r["task"]=="need" and r["representation"] in REPS]
    mech={norm(r["sha256"]):r for r in read(MECH)};fail={norm(r["sha256"]):r for r in read(FAIL)}
    routed={sha:r for sha,r in mech.items() if r["cohort"] in ("C","D","E","F")}
    if len(routed)!=497 or Counter(r["cohort"] for r in routed.values())!=Counter({"E":360,"D":74,"C":52,"F":11}):
        raise RuntimeError("BLOCKED_BY_CONTRACT: locked cohort counts changed")
    fc=Counter(r["failure_type"] for r in fail.values())
    if fc!=Counter({"B_RIGHT_DIRECTION_TOO_WEAK":29,"A_WRONG_DIRECTION":27,"C_BOUND_LIMIT":14,"D_NEAR_ZERO":4}):
        raise RuntimeError("BLOCKED_BY_CONTRACT: locked failure counts changed: "+str(fc))
    score=defaultdict(dict)
    for r in predictions:
        sha=norm(r["sha256"]);rep=r["representation"]
        if rep in score[sha]:raise RuntimeError("duplicate OOF gate score")
        score[sha][rep]=float(r["probability"])
    if set(score)!=set(routed) or any(set(v)!=set(REPS) for v in score.values()):raise RuntimeError("OOF gate score/cohort mismatch")
    names={"C":"FIXED","D":"ROUTED_UNFIXED","E":"PRESERVED","F":"BROKEN"};rows=[]
    for sha,m in routed.items():
        failure=fail.get(sha,{}).get("failure_type","")
        rows.append({"sha256":sha,"year":int(m["year"]),"fold":int(m["fold"]),"label":int(m["label"]),
          "cohort":m["cohort"],"cohort_name":names[m["cohort"]],"failure_type":failure,
          "stage1_wrong":int(m["cohort"] in ("C","D")),"historical_M1_fixed":int(m["cohort"]=="C"),
          "historical_M1_broken":int(m["cohort"]=="F"),**{rep:score[sha][rep] for rep in REPS},
          **{rep+"__gate_at_0p5":int(score[sha][rep]>=.5) for rep in REPS}})
    write(OUT/"historical_cohort_gate_scores.csv",rows)
    summary=[]
    groups=[("C_FIXED",[r for r in rows if r["cohort"]=="C"]),("D_ROUTED_UNFIXED",[r for r in rows if r["cohort"]=="D"]),
            ("E_PRESERVED",[r for r in rows if r["cohort"]=="E"]),("F_BROKEN",[r for r in rows if r["cohort"]=="F"])]
    for kind in ("A_WRONG_DIRECTION","B_RIGHT_DIRECTION_TOO_WEAK","C_BOUND_LIMIT","D_NEAR_ZERO"):
        groups.append((kind,[r for r in rows if r["failure_type"]==kind]))
    for group,items in groups:
      for rep in REPS:summary.append({"group":group,"representation":rep,"n":len(items),**stats([r[rep] for r in items])})
    write(OUT/"gate_score_summary_by_historical_group.csv",summary)
    year=[]
    for y in (2018,2019,2020,2021):
      for group,items0 in groups:
       items=[r for r in items0 if r["year"]==y]
       if not items:continue
       for rep in REPS:year.append({"year":y,"group":group,"representation":rep,"n":len(items),**stats([r[rep] for r in items])})
    write(OUT/"gate_score_summary_by_year.csv",year)

    comparisons=[]
    named={name:items for name,items in groups}
    pairs=(("C_FIXED","F_BROKEN","repairable-vs-harmful"),("C_FIXED","E_PRESERVED","fixed-vs-preserved"),
           ("D_ROUTED_UNFIXED","E_PRESERVED","unfixed-vs-preserved"),("A_WRONG_DIRECTION","B_RIGHT_DIRECTION_TOO_WEAK","wrong-direction-vs-too-weak"))
    for a,b,label in pairs:
      for rep in REPS:
        av=[r[rep] for r in named[a]];bv=[r[rep] for r in named[b]]
        target=np.asarray([1]*len(av)+[0]*len(bv));value=np.asarray(av+bv,float)
        comparisons.append({"comparison":label,"positive_group":a,"negative_group":b,"representation":rep,
          "n_positive":len(av),"n_negative":len(bv),"score_AUC":float(roc_auc_score(target,value)),**bootstrap_difference(av,bv)})
    write(OUT/"historical_group_score_comparisons.csv",comparisons)

    threshold=[]
    for rep in REPS:
      for group,items in groups:
        threshold.append({"representation":rep,"group":group,"n":len(items),"gate_positive_n":sum(r[rep]>=.5 for r in items),
                          "gate_positive_rate":float(np.mean([r[rep]>=.5 for r in items]))})
    write(OUT/"fixed_0p5_gate_capture.csv",threshold)
    main_rep="S3_tail88_plus_logit";full_rep="S4_full_R4"
    sm={(r["group"],r["representation"]):r for r in summary};cmp={(r["comparison"],r["representation"]):r for r in comparisons}
    lines=["CARE-Droid historical cohort gate mapping audit","Read-only post-hoc analysis of existing strict OOF need-correction probabilities.",
      "No threshold search; 0.5 is the probe's fixed decision threshold. Cohorts remain anchored to locked historical M1 outcomes.","",
      f"Locked counts: Fixed=52, RoutedUnfixed=74, Preserved=360, Broken=11; WrongDirection=27, TooWeak=29, BoundLimit=14, NearZero=4.","",
      f"S3 gate capture Fixed={sm[('C_FIXED',main_rep)]['threshold_0p5_positive_n']}/52 ({sm[('C_FIXED',main_rep)]['threshold_0p5_positive_rate']:.3%})",
      f"S3 gate capture WrongDirection={sm[('A_WRONG_DIRECTION',main_rep)]['threshold_0p5_positive_n']}/27 ({sm[('A_WRONG_DIRECTION',main_rep)]['threshold_0p5_positive_rate']:.3%})",
      f"S3 gate capture TooWeak={sm[('B_RIGHT_DIRECTION_TOO_WEAK',main_rep)]['threshold_0p5_positive_n']}/29 ({sm[('B_RIGHT_DIRECTION_TOO_WEAK',main_rep)]['threshold_0p5_positive_rate']:.3%})",
      f"S3 gate activation Broken={sm[('F_BROKEN',main_rep)]['threshold_0p5_positive_n']}/11 ({sm[('F_BROKEN',main_rep)]['threshold_0p5_positive_rate']:.3%})",
      f"S3 gate activation Preserved={sm[('E_PRESERVED',main_rep)]['threshold_0p5_positive_n']}/360 ({sm[('E_PRESERVED',main_rep)]['threshold_0p5_positive_rate']:.3%})","",
      f"S4 Fixed-vs-Broken score AUC={cmp[('repairable-vs-harmful',full_rep)]['score_AUC']:.6f}; mean-score-difference CI=[{cmp[('repairable-vs-harmful',full_rep)]['ci_low']:.6f},{cmp[('repairable-vs-harmful',full_rep)]['ci_high']:.6f}]",
      f"S4 WrongDirection-vs-TooWeak score AUC={cmp[('wrong-direction-vs-too-weak',full_rep)]['score_AUC']:.6f}.","",
      "Interpretation boundary: this audit evaluates whether an already-trained diagnostic probe score aligns with historical cohorts. It does not estimate a new CARE-Droid model, correction policy, or deployable gate."]
    (OUT/"HISTORICAL_GATE_COHORT_REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    write_json(OUT/"NO_MODEL_MODIFICATION_AUDIT.json",{"model_loaded":False,"stage2_forward_invoked":False,"training_invoked":False,
      "threshold_search_invoked":False,"fixed_threshold":0.5,"locked_cohort_counts":{"C":52,"D":74,"E":360,"F":11},
      "probe_scores_source":str(PRED),"historical_outcomes_source":str(MECH)})
    print("HISTORICAL_GATE_COHORT_AUDIT_DONE")

if __name__=="__main__":main()
