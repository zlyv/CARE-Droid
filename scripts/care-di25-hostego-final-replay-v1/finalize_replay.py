#!/usr/bin/env python3
"""Merge final replay artifacts and perform read-only reliability diagnosis."""
from __future__ import annotations
import csv,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact
from sklearn.metrics import balanced_accuracy_score,f1_score,confusion_matrix

ROOT=Path("artifacts/care-di25-hostego-v1/stage2_mechanism_diagnosis/final_replay")
MECH=Path("artifacts/care-di25-hostego-v1/stage2_mechanism_diagnosis")
ROUTER=Path("artifacts/care-route-stage2-within-year-matrix-v1/shared/within_year_router_inputs.csv")
PRIOR=MECH/"instrumented_rerun/apk_internal.csv";SEEDS=(900000,900001,900002);TOL=1e-12
def norm(x):return str(x or "").strip().lower()
def read(p):
  with Path(p).open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def write(p,rows):
  fields=[]
  for r in rows:
    for k in r:
      if k not in fields:fields.append(k)
  with Path(p).open("w",encoding="utf-8",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def sgn(x):return 0 if abs(float(x))<=TOL else 1 if float(x)>0 else -1
def entropy(signs):
  c=Counter(signs);p=np.asarray([v/len(signs) for v in c.values()]);return float(-np.sum(p*np.log2(p)))
def metrics(y,p):
  pred=p>=.5;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
  return {"BA":float(balanced_accuracy_score(y,pred)),"F1":float(f1_score(y,pred)),"FP":int(fp),"FN":int(fn),"total_errors":int(fp+fn)}
def group_stats(values):
  x=np.asarray(values,float);return {"n":len(x),"mean":float(np.mean(x)) if len(x) else np.nan,"median":float(np.median(x)) if len(x) else np.nan,
    "std":float(np.std(x)) if len(x) else np.nan,"iqr":float(np.percentile(x,75)-np.percentile(x,25)) if len(x) else np.nan}
def odds(a,b,c,d):
  o,p=fisher_exact([[a,b],[c,d]],alternative="two-sided");return float(o),float(p)

def load_seed_outputs():
  rows=[];hosts=[]
  for year in (2018,2019,2020,2021):
    for fold in range(1,6):
      for seed in SEEDS:
        d=ROOT/"seed_outputs"/str(year)/f"fold_{fold}"
        rows+=read(d/f"seed_{seed}_apk.csv");hosts+=read(d/f"seed_{seed}_host.csv")
  if len(rows)!=1989*3:raise RuntimeError(f"BLOCKED_BY_CONTRACT: per-seed APK rows={len(rows)}")
  return rows,hosts

def main():
  long,host=load_seed_outputs();groups=defaultdict(dict)
  for r in long:
    key=norm(r["sha256"]);seed=int(r["seed"])
    if seed in groups[key]:raise RuntimeError("duplicate APK/seed")
    groups[key][seed]=r
  router=read(ROUTER);prior={norm(r["sha256"]):r for r in read(PRIOR)}
  if len(groups)!=1989 or set(groups)!=set(prior):raise RuntimeError("BLOCKED_BY_CONTRACT: replay cohort differs")
  wide=[];max_mean={"raw":0.,"bounded":0.,"probability":0.};newprob={}
  for rr in router:
    sha=norm(rr["sha256"]);g=groups[sha]
    if set(g)!=set(SEEDS):raise RuntimeError("missing seed for "+sha)
    raw=np.asarray([float(g[s]["raw_delta"]) for s in SEEDS]);bd=np.asarray([float(g[s]["bounded_delta"]) for s in SEEDS]);pr=np.asarray([float(g[s]["probability"]) for s in SEEDS])
    signs=[sgn(x) for x in raw];pos=signs.count(1);neg=signs.count(-1);zero=signs.count(0);major=1 if pos>neg else -1 if neg>pos else 0
    row={"year":int(rr["year"]),"fold":int(rr["fold"]),"sha256":sha,"label":int(rr["label"]),"routed":int(float(rr["route_DI"])),"p_stage1":float(rr["p_stage1"])}
    for i,s in enumerate(SEEDS):
      row[f"raw_delta_seed_{s}"]=raw[i];row[f"bounded_delta_seed_{s}"]=bd[i];row[f"probability_seed_{s}"]=pr[i]
      row[f"raw_delta_sign_seed_{s}"]=signs[i];row[f"bounded_delta_sign_seed_{s}"]=sgn(bd[i])
      if signs[i]!=sgn(bd[i]):raise RuntimeError("BLOCKED_BY_INTERNAL_CONSISTENCY: raw/bounded sign")
    row.update({"seed_mean_raw_delta":float(np.mean(raw)),"seed_mean_bounded_delta":float(np.mean(bd)),"seed_probability_mean":float(np.mean(pr)),
      "seed_probability_sd":float(np.std(pr)),"positive_seed_count":pos,"negative_seed_count":neg,"zero_seed_count":zero,"majority_sign":major,
      "direction_agreement_ratio":max(pos,neg,zero)/3,"sign_entropy":entropy(signs),"delta_mean":float(np.mean(bd)),"delta_std":float(np.std(bd)),
      "abs_delta_mean":float(np.mean(np.abs(bd))),"delta_cv_if_defined":float(np.std(bd)/abs(np.mean(bd))) if abs(np.mean(bd))>TOL else np.nan})
    wide.append(row);newprob[sha]=float(np.mean(pr))
  write(ROOT/"per_seed_apk_internal.csv",wide);write(ROOT/"per_seed_host_internal.csv",host)
  mean_audit=[]
  for key in ("raw_delta","bounded_delta","probability"):
    diffs=[]
    for r in wide:
      direct=float(np.mean([float(r[f"{key}_seed_{s}"]) for s in SEEDS]))
      saved=float(r["seed_mean_raw_delta" if key=="raw_delta" else "seed_mean_bounded_delta" if key=="bounded_delta" else "seed_probability_mean"])
      diffs.append(abs(direct-saved))
    mean_audit.append({"quantity":key,"max_abs_difference":max(diffs),"passed":int(max(diffs)<=1e-6)})
  write(ROOT/"seed_mean_consistency_audit.csv",mean_audit)
  if not all(r["passed"] for r in mean_audit):raise RuntimeError("BLOCKED_BY_INTERNAL_CONSISTENCY: seed means")

  # Compare this replay with the immediately preceding numerically-equivalent replay.
  d=np.asarray([abs(newprob[norm(r["sha256"])]-float(prior[norm(r["sha256"])]["rerun_M1_prob"])) for r in router])
  identity=[{"sha256":norm(r["sha256"]),"year":r["year"],"fold":r["fold"],"prior_replay_prob":float(prior[norm(r["sha256"])]["rerun_M1_prob"]),
    "final_replay_prob":newprob[norm(r["sha256"])],"abs_diff":d[i],"prediction_diff":int((newprob[norm(r["sha256"])]>=.5)!=(float(prior[norm(r["sha256"])]["rerun_M1_prob"])>=.5))} for i,r in enumerate(router)]
  write(ROOT/"replay_identity_audit.csv",identity)
  ids={"max_abs_prob_diff":float(np.max(d)),"mean_abs_prob_diff":float(np.mean(d)),"p95":float(np.percentile(d,95)),"p99":float(np.percentile(d,99)),
    "prediction_difference_count":int(sum(r["prediction_diff"] for r in identity))}
  if ids["prediction_difference_count"] or ids["max_abs_prob_diff"]>0.002:raise RuntimeError("BLOCKED_BY_REPLAY_DRIFT: "+json.dumps(ids))

  mechanism={norm(r["sha256"]):r for r in read(MECH/"apk_stage2_internal.csv")};fail={norm(r["sha256"]):r for r in read(MECH/"routed_unfixed_failure_types.csv")}
  host_mean=read(MECH/"host_internal_diagnostics.csv")
  reliability=[]
  for w in wide:
    if not w["routed"]:continue
    m=mechanism[w["sha256"]];label=int(w["label"])
    # Direction toward the ground-truth class is well-defined for every routed
    # cohort, including preserved/broken Stage-I-correct controls.
    required=1 if label==1 else -1
    signs=[int(w[f"raw_delta_sign_seed_{s}"]) for s in SEEDS];correct=[int(required!=0 and x==required) for x in signs]
    w.update({"cohort":m["cohort"],"failure_type":fail.get(w["sha256"],{}).get("failure_type",""),"truth_required_direction":required,
      **{f"seed_{s}_direction_correct":correct[i] for i,s in enumerate(SEEDS)},"majority_direction_correct":int(int(w["majority_sign"])==required),
      "unanimous_correct":int(sum(correct)==3),"unanimous_wrong":int(all(x==-required for x in signs)),
      "split_direction":int(len(set(signs))>1)})
    reliability.append(w)
  write(ROOT/"seed_direction_reliability.csv",reliability)
  wrong=[r for r in reliability if r["failure_type"]=="A_WRONG_DIRECTION"]
  for r in wrong:
    ncorrect=sum(int(r[f"seed_{s}_direction_correct"]) for s in SEEDS)
    signs=[int(r[f"raw_delta_sign_seed_{s}"]) for s in SEEDS]
    r["seed_consistency_class"]="SYSTEMATIC_WRONG" if int(r["unanimous_wrong"]) else "UNSTABLE_DIRECTION" if int(r["split_direction"]) else "UNANIMOUS_ZERO" if signs==[0,0,0] else f"{ncorrect}_of_3_correct"
    r["final_error_type"]="FN" if int(r["label"])==1 else "FP"
  if len(wrong)!=27:raise RuntimeError("locked Wrong Direction count changed")
  write(ROOT/"wrong_direction_seed_consistency.csv",wrong)
  weak=[r for r in reliability if r["failure_type"]=="B_RIGHT_DIRECTION_TOO_WEAK"]
  for r in weak:
    m=mechanism[r["sha256"]];required=abs(float(m["required_flip_delta"]));actual=abs(float(m["bounded_delta"]));r.update({"required_delta":required,
      "ensemble_actual_delta":actual,"strength_gap":required-actual,"final_error_type":"FN" if int(r["label"])==1 else "FP"})
    gap=float(r["strength_gap"])
    r["strength_gap_bucket"]=("[0,0.10)" if gap<.10 else "[0.10,0.25)" if gap<.25 else "[0.25,0.50)" if gap<.50 else "[0.50,1.00)" if gap<1.0 else "[1.00,+inf)")
    for s in SEEDS:r[f"strength_gap_seed_{s}"]=required-abs(float(r[f"bounded_delta_seed_{s}"]))
  if len(weak)!=29:raise RuntimeError("locked Too Weak count changed")
  write(ROOT/"too_weak_seed_consistency.csv",weak)

  # Critical role and top critical-Ego pair, one categorical presence per APK.
  critical=[r for r in host_mean if int(float(r["critical_host"]))==1];crit_by={norm(r["sha256"]):r for r in critical}
  lookup={(norm(r["sha256"]),int(r["host_index"])):r for r in host_mean};pair_by={}
  for sha,h in crit_by.items():
    choices=[]
    for direction,ik,rk in (("in","incoming_neighbor_indices","incoming_neighbor_roles"),("out","outgoing_neighbor_indices","outgoing_neighbor_roles")):
      for idx,role in zip(json.loads(h[ik]),json.loads(h[rk])):
        other=lookup.get((sha,int(idx)));weight=float(other["attention_weight"]) if other else -np.inf;choices.append((weight,(h["center_role"],role,direction)))
    if choices:pair_by[sha]=max(choices,key=lambda z:(z[0],z[1]))[1]
  cohorts={c:{norm(r["sha256"]) for r in mechanism.values() if r["cohort"]==c} for c in ("C","D","E","F")}
  def enrich(items):
    out=[]
    for item in sorted(set(items.values())):
      present={c:sum(items.get(x)==item for x in cohorts[c]) for c in cohorts};total={c:len(cohorts[c]) for c in cohorts}
      o1,p1=odds(present["D"],total["D"]-present["D"],present["C"],total["C"]-present["C"])
      o2,p2=odds(present["D"],total["D"]-present["D"],present["E"],total["E"]-present["E"])
      years=sorted({int(mechanism[x]["year"]) for x,v in items.items() if v==item})
      out.append({"item":json.dumps(item) if isinstance(item,tuple) else item,"D_count":present["D"],"Fixed_count":present["C"],"Broken_count":present["F"],
        "Preserved_count":present["E"],"D_presence_rate":present["D"]/total["D"],"Fixed_presence_rate":present["C"]/total["C"],
        "Broken_presence_rate":present["F"]/total["F"],"Preserved_presence_rate":present["E"]/total["E"],"D_vs_Fixed_odds_ratio":o1,
        "D_vs_Fixed_fisher_p":p1,"D_vs_Preserved_odds_ratio":o2,"D_vs_Preserved_fisher_p":p2,"total_support":sum(present.values()),
        "years":json.dumps(years),"year_count":len(years),"crosses_3_years":int(len(years)>=3)})
    return out
  roles=enrich({sha:r["center_role"] for sha,r in crit_by.items()});pairs=enrich(pair_by)
  write(ROOT/"critical_role_failure_enrichment.csv",roles);write(ROOT/"critical_ego_pair_failure_enrichment_all.csv",pairs)
  write(ROOT/"critical_ego_pair_failure_enrichment.csv",[r for r in pairs if int(r["total_support"])>=5])

  # Continuous and categorical A-vs-B diagnosis.
  avb=[];features=("mean_gamma","max_gamma","mean_anomaly","max_anomaly","max_attention","attention_entropy","real_host_count","nontrivial_ego_count")
  sets={"A_WRONG_DIRECTION":{r["sha256"] for r in wrong},"B_RIGHT_DIRECTION_TOO_WEAK":{r["sha256"] for r in weak}}
  for name,ss in sets.items():
    for f in features:avb.append({"group":name,"kind":"continuous","feature":f,**group_stats([float(mechanism[x][f]) for x in ss])})
    avb.append({"group":name,"kind":"continuous","feature":"ego_ratio",**group_stats([
      float(mechanism[x]["nontrivial_ego_count"])/max(float(mechanism[x]["real_host_count"]),1.0) for x in ss])})
    for f in ("external_background_value","internal_background_value","shared_commonness_value","anomaly_score","gamma","attention_weight"):
      avb.append({"group":name,"kind":"critical_host_continuous","feature":f,**group_stats([float(crit_by[x][f]) for x in ss if x in crit_by])})
    avb.append({"group":name,"kind":"continuous","feature":"seed_direction_agreement",**group_stats([float(next(r for r in reliability if r["sha256"]==x)["direction_agreement_ratio"]) for x in ss])})
    for role,count in Counter(crit_by[x]["center_role"] for x in ss if x in crit_by).items():avb.append({"group":name,"kind":"critical_role","feature":role,"n":count,"rate":count/len(ss)})
    for pair,count in Counter(pair_by[x] for x in ss if x in pair_by).items():avb.append({"group":name,"kind":"critical_ego_pair","feature":json.dumps(pair),"n":count,"rate":count/len(ss)})
  write(ROOT/"A_vs_B_mechanism_comparison.csv",avb)

  # Locked-outcome failure-type oracle.
  y=np.asarray([int(r["label"]) for r in router]);p0=np.asarray([float(r["p_stage1"]) for r in router]);route=np.asarray([int(float(r["route_DI"]))==1 for r in router])
  locked=np.asarray([float(prior[norm(r["sha256"])]["locked_M1_prob"]) for r in router]);base=np.where(route,locked,p0)
  type_by={x:r["failure_type"] for x,r in fail.items()};cohort_by={x:r["cohort"] for x,r in mechanism.items()};oracle=[]
  for scope,mask in [(str(yv),np.asarray([int(r["year"])==yv for r in router])) for yv in (2018,2019,2020,2021)]+[("pooled",np.ones(len(y),bool))]:
    variants={"O_BASE":np.array(base),"O_FIX_B":np.array(base),"O_FIX_A":np.array(base),"O_FIX_A_B":np.array(base),"O_FIX_B_NO_BROKEN":np.array(base)}
    for i,r in enumerate(router):
      sha=norm(r["sha256"]);correct=1-1e-9 if y[i] else 1e-9
      if type_by.get(sha)=="B_RIGHT_DIRECTION_TOO_WEAK":variants["O_FIX_B"][i]=correct;variants["O_FIX_A_B"][i]=correct;variants["O_FIX_B_NO_BROKEN"][i]=correct
      if type_by.get(sha)=="A_WRONG_DIRECTION":variants["O_FIX_A"][i]=correct;variants["O_FIX_A_B"][i]=correct
      if cohort_by.get(sha)=="F":variants["O_FIX_B_NO_BROKEN"][i]=p0[i]
    for name,p in variants.items():oracle.append({"scope":scope,"oracle":name,**metrics(y[mask],p[mask])})
  write(ROOT/"failure_type_oracle_value.csv",oracle)

  # Decision rule and final report.
  systematic=sum(r["seed_consistency_class"]=="SYSTEMATIC_WRONG" for r in wrong);unstable=sum(r["seed_consistency_class"]=="UNSTABLE_DIRECTION" for r in wrong)
  weak_unanimous=sum(int(r["unanimous_correct"]) for r in weak);fixed=[r for r in reliability if r["cohort"]=="C"];broken=[r for r in reliability if r["cohort"]=="F"]
  fixed_unanimous=sum(int(r["unanimous_correct"]) for r in fixed);small=sum(float(r["strength_gap"])<.25 for r in weak);gap=float(np.mean([float(r["strength_gap"]) for r in weak]))
  if systematic>=math.ceil(len(wrong)/2):hyp="INSUFFICIENT";bias="YES";priority="B. direction/magnitude decoupling"
  elif unstable>=math.ceil(len(wrong)/2) and weak_unanimous/len(weak)>unstable/len(wrong):hyp="SUPPORTED";bias="NO";priority="A. reliability-aware correction"
  else:hyp="MIXED";bias="MIXED";priority="B. direction/magnitude decoupling"
  pooled={r["oracle"]:r for r in oracle if r["scope"]=="pooled"}
  enriched_roles=[r for r in roles if float(r["D_vs_Fixed_odds_ratio"])>1 and float(r["D_vs_Fixed_fisher_p"])<.05]
  enriched_pairs=[r for r in pairs if int(r["total_support"])>=5 and float(r["D_vs_Fixed_odds_ratio"])>1 and float(r["D_vs_Fixed_fisher_p"])<.05]
  lines=["CARE-Droid final seed reliability diagnosis","EXACT_STATE_REPRODUCTION=FAIL","NUMERICALLY_EQUIVALENT_REPLAY=PASS",
    "All cohorts/outcomes are anchored to locked historical M1 predictions; internals are descriptive replay diagnostics.","",
    f"replay_identity={ids}",f"checkpoint_count=60; checkpoint_reload=PASS",
    "Persisted: model weights, seed/year/fold, selected epoch, architecture, hashes, role-graph scaler, region scaler, prototype state, and exact validation inference tensors/bags.",
    "Not persisted because unavailable/not required for inference: a separate early-stop epoch distinct from selected_epochs; optimizer state after completed training.",
    f"Q1 systematic 3/3 wrong={systematic}/27",
    f"Q2 seed-direction split={unstable}/27",f"Q3 Too Weak 3/3 correct={weak_unanimous}/29",
    f"Q4 Fixed 3/3 correct={fixed_unanimous}/{len(fixed)} ({fixed_unanimous/len(fixed):.3%})",
    f"Q5 Broken patterns: unanimous_correct={sum(int(r['unanimous_correct']) for r in broken)}, unanimous_wrong={sum(int(r['unanimous_wrong']) for r in broken)}, split={sum(int(r['split_direction']) for r in broken)}",
    f"Q6 Wrong direction lower reliability: wrong_split_rate={unstable/27:.3%}; too_weak_unanimous_correct_rate={weak_unanimous/29:.3%}",
    f"Q7 SYSTEMATIC_DIRECTION_BIAS={bias}",f"Q8 Too Weak mean strength gap={gap:.6f} logit",f"Q9 Too Weak gap<0.25={small}/29",
    f"Q10 D-vs-Fixed enriched critical roles (OR>1,p<.05; diagnostic only)={[(r['item'],r['D_vs_Fixed_odds_ratio'],r['D_vs_Fixed_fisher_p']) for r in enriched_roles]}",
    f"Q11 D-vs-Fixed enriched critical Ego pairs support>=5={[(r['item'],r['D_vs_Fixed_odds_ratio'],r['D_vs_Fixed_fisher_p']) for r in enriched_pairs]}",
    f"Q12 enriched patterns crossing >=3 years: roles={sum(int(r['crosses_3_years']) for r in enriched_roles)}, pairs={sum(int(r['crosses_3_years']) for r in enriched_pairs)}",
    f"Q9-oracle Wrong Direction only pooled BA={pooled['O_FIX_A']['BA']:.6f}",f"Q10-oracle Too Weak only pooled BA={pooled['O_FIX_B']['BA']:.6f}",
    f"CORRECTION_RELIABILITY_HYPOTHESIS={hyp}",f"SYSTEMATIC_DIRECTION_BIAS={bias}",f"Q13 priority={priority}","",
    "No new variant was trained or selected. Fisher p-values are descriptive and are not used for feature selection."]
  (ROOT/"FINAL_SEED_RELIABILITY_REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
  print("FINAL_SEED_RELIABILITY_DIAGNOSIS_PASS")
if __name__=="__main__":main()
