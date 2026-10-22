#!/usr/bin/env python3
from __future__ import annotations
import math
import numpy as np
from scipy.stats import mannwhitneyu,spearmanr
from sklearn.metrics import roc_auc_score
from common import ROOT,BASE,METHOD,MAX_DELTA,read_csv,write_csv,dump,logit,norm

src=read_csv(BASE);rows=[]
diag=("critical_score","mean_instance_score","attention_entropy","max_attention","context_entropy","mean_external_distance","mean_internal_distance","mean_shared_commonness","max_local_residual","graph_pool_entropy")
for r in src:
    if not int(r["A25__routed"]):continue
    y=int(r["label"]);base=float(r["Global24__prob"]);expert=float(r[METHOD+"__expert_prob"]);final=float(r[METHOD+"__prob"]);delta=float(r[METHOD+"__bounded_delta"])
    if abs(final-expert)>1e-8:raise RuntimeError("BLOCKED_BY_CONTRACT: routed final != C43 candidate")
    bp=int(base>=.5);cp=int(expert>=.5);base_ok=bp==y;cand_ok=cp==y
    if not base_ok and cand_ok:out="FIXED"
    elif base_ok and not cand_ok:out="BROKEN"
    elif base_ok:out="UNCHANGED_CORRECT"
    else:out="UNCHANGED_WRONG"
    sm=abs(float(logit([base])[0]));effective=float(logit([expert])[0]-logit([base])[0]);opp=delta if bp==0 else -delta
    q={"sha256":norm(r["sha256"]),"fold":int(r["fold"]),"true_label":y,"stage1_pred":bp,"stage1_prob":base,"stage1_margin":sm,"router_score":float(r["A25__score"]),"routed":1,"c43_raw_score":"","c43_raw_score_available":0,"c43_raw_score_provenance":"not saved; A0 averages three seed diagnostics so raw_delta cannot be recovered from their mean bounded_delta","c43_candidate_prob":expert,"c43_candidate_pred":cp,"raw_delta":"","raw_delta_available":0,"raw_delta_provenance":"not saved and not exactly reconstructable after three-seed ensemble averaging","bounded_delta":delta,"bounded_delta_provenance":"saved mean over three seed-level bounded deltas","abs_raw_delta":"","abs_bounded_delta":abs(delta),"effective_ensemble_delta":effective,"abs_effective_ensemble_delta":abs(effective),"delta_sign":int(np.sign(delta)),"opposition_strength":opp,"abs_probability_shift":abs(expert-base),"candidate_margin":abs(float(logit([expert])[0])),"correction_excess_over_stage1_margin":opp-sm,"prediction_changed":int(bp!=cp),"final_prob":final,"final_pred":cp,"outcome":out}
    for name in diag:
        k=METHOD+"__"+name
        if k in r and r[k] != "":q[name]=float(r[k])
    if not np.isfinite(np.asarray([v for v in q.values() if isinstance(v,(int,float))],float)).all():raise RuntimeError("NaN/Inf anatomy")
    rows.append(q)
if len(rows)!=200:raise RuntimeError("BLOCKED_BY_CONTRACT: routed anatomy is not 200")
write_csv(ROOT/"d0_correction_score_anatomy.csv",rows)

changed=[r for r in rows if r["outcome"] in {"FIXED","BROKEN"}]
if sum(r["outcome"]=="FIXED" for r in changed)!=11 or sum(r["outcome"]=="BROKEN" for r in changed)!=7:raise RuntimeError("D0 changed cohort mismatch")
score_specs=[
 ("abs_bounded_delta","abs(saved bounded correction)",True),
 ("abs_effective_ensemble_delta","abs(logit(ensemble candidate probability)-logit(Stage-I probability))",True),
 ("abs_probability_shift","abs(C43 candidate probability - Stage-I probability)",True),
 ("candidate_margin","abs(logit(C43 candidate probability))",True),
 ("opposition_strength","bounded_delta if Stage-I predicts benign else -bounded_delta",True),
 ("correction_excess_over_stage1_margin","opposition_strength - abs(Stage-I logit)",True),
]
for name in diag:
    if name in rows[0]:score_specs.append((name,"saved A0 inference diagnostic field",True))
audit=[];fold_audit=[]
for name,formula,deployable in score_specs:
    a=np.asarray([float(r[name]) for r in changed]);t=np.asarray([r["outcome"]=="FIXED" for r in changed],int);fx=a[t==1];br=a[t==0];auc=float(roc_auc_score(t,a));u,p=mannwhitneyu(fx,br,alternative="two-sided")
    audit.append({"score":name,"formula":formula,"label_free":int(deployable),"deployable":int(deployable),"fixed_n":len(fx),"broken_n":len(br),"fixed_mean":fx.mean(),"fixed_std":fx.std(),"fixed_median":np.median(fx),"fixed_q25":np.quantile(fx,.25),"fixed_q75":np.quantile(fx,.75),"broken_mean":br.mean(),"broken_std":br.std(),"broken_median":np.median(br),"broken_q25":np.quantile(br,.25),"broken_q75":np.quantile(br,.75),"raw_auc_fixed_positive":auc,"mann_whitney_u":float(u),"p_value":float(p),"rank_biserial_effect":2*auc-1})
    for f in range(1,6):
        z=[r for r in changed if int(r["fold"])==f];zt=np.asarray([r["outcome"]=="FIXED" for r in z],int);za=np.asarray([float(r[name]) for r in z])
        defined=len(set(zt.tolist()))==2;fa=float(roc_auc_score(zt,za)) if defined else None
        fold_audit.append({"score":name,"fold":f,"fixed_n":int(zt.sum()),"broken_n":int(len(zt)-zt.sum()),"auc_status":"DEFINED" if defined else "UNDEFINED_DUE_TO_SINGLE_CLASS","raw_auc_fixed_positive":"" if fa is None else fa,"direction_consistent_with_higher_is_fixed":"" if fa is None else int(fa>.5)})
write_csv(ROOT/"d0_score_fixed_broken_audit.csv",audit);write_csv(ROOT/"d0_score_fixed_broken_fold_audit.csv",fold_audit)

# Strength/coverage curve is an audit, not a gate: rank every routed D0
# candidate by its pre-prediction, label-free correction magnitude.
ordered=sorted(rows,key=lambda r:(-float(r["abs_bounded_delta"]),r["sha256"]));curve=[]
for pct in range(10,101,10):
    n=int(math.ceil(len(ordered)*pct/100));z=ordered[:n];fx=sum(r["outcome"]=="FIXED" for r in z);br=sum(r["outcome"]=="BROKEN" for r in z)
    curve.append({"top_percent":pct,"n":n,"fixed":fx,"broken":br,"net":fx-br,"prediction_changed":fx+br,"benefit_rate_among_changed":fx/max(fx+br,1),"mean_abs_delta":float(np.mean([r["abs_bounded_delta"] for r in z])),"min_abs_delta":float(z[-1]["abs_bounded_delta"])})
write_csv(ROOT/"d0_score_strength_curve.csv",curve)

primary=next(r for r in audit if r["score"]=="abs_bounded_delta");pf=[r for r in fold_audit if r["score"]=="abs_bounded_delta"];eligible=[r for r in pf if r["auc_status"]=="DEFINED"];consistent=sum(int(r["direction_consistent_with_higher_is_fixed"]) for r in eligible)
fxscore=np.asarray([r["abs_bounded_delta"] for r in changed]);order=np.argsort(-fxscore);half=max(1,len(order)//2);top=[changed[i] for i in order[:half]];bottom=[changed[i] for i in order[half:]]
top_rate=sum(r["outcome"]=="FIXED" for r in top)/len(top);bottom_rate=sum(r["outcome"]=="FIXED" for r in bottom)/len(bottom)
rho=float(spearmanr([r["abs_bounded_delta"] for r in rows],[-r["stage1_margin"] for r in rows]).statistic);rho=0. if not np.isfinite(rho) else rho
stable=(bool(primary["deployable"]) and float(primary["raw_auc_fixed_positive"])>=.65 and len(eligible)>=3 and consistent>len(eligible)/2 and top_rate>bottom_rate and abs(rho)<.80)
provenance={
 "score_name":"abs_bounded_delta","score_formula":"abs(mean_seed(2.25*tanh(raw_delta_seed)))","source_csv_field":METHOD+"__bounded_delta",
 "generated_before_final_prediction":True,"available_at_inference":True,"label_free":True,"uses_true_label_to_orient_score":False,"uses_fixed_broken_to_construct_score":False,"uses_correctness_or_error_type_to_construct_score":False,
 "whole_799_statistics_used":False,"outer_test_used_for_normalization":False,"future_rows_loaded":0,"strict_outer_oof":True,
 "auc_cohort":"D0 prediction-changed samples only: FIXED union BROKEN","positive_class":"FIXED","auc_is_raw_not_max_flipped":True,
 "fixed_n":11,"broken_n":7,"pooled_raw_auc":float(primary["raw_auc_fixed_positive"]),"eligible_folds":len(eligible),"consistent_folds_higher_is_fixed":consistent,
 "fold_results":pf,"top_half_changed_benefit_rate":top_rate,"bottom_half_changed_benefit_rate":bottom_rate,
 "spearman_abs_delta_vs_stage1_uncertainty":rho,"stability_gate_passed":stable,
 "raw_delta_available":False,"raw_delta_reason":"The saved A0 result averages three seed-level bounded deltas and probabilities; arctanh(mean bounded delta / 2.25) is not the mean raw delta and is therefore not reported.",
 "important_equivalence":"On prediction-changed samples, correction necessarily opposes Stage-I, so the earlier opposition-strength AUC and abs_bounded_delta AUC are identical. No true label is used for this equality."
}
dump(ROOT/"d0_auc_provenance_audit.json",provenance);dump(ROOT/"strength_signal_decision.json",{"supported":stable,"decision":"D0_CORRECTION_STRENGTH_SIGNAL_SUPPORTED" if stable else "D0_CORRECTION_STRENGTH_SIGNAL_NOT_STABLE","rule":"pooled raw AUC>=.65, >=3 eligible folds, strict majority raw AUC>.5, top-half enrichment, |rho with Stage-I uncertainty|<.80","future_rows_loaded":0})
print("D0_0805_SCORE_NAME=abs_bounded_delta");print("D0_0805_SCORE_FORMULA=abs(mean_seed(2.25*tanh(raw_delta_seed)))");print("D0_0805_LABEL_FREE=true");print("D0_0805_DEPLOYABLE=true");print(f"D0_0805_POOLED_AUC={float(primary['raw_auc_fixed_positive']):.12f}")
for r in pf:print(f"fold{r['fold']}: fixed={r['fixed_n']} broken={r['broken_n']} auc={r['raw_auc_fixed_positive'] if r['auc_status']=='DEFINED' else r['auc_status']}")
print("D0_CORRECTION_STRENGTH_SIGNAL_SUPPORTED" if stable else "D0_CORRECTION_STRENGTH_SIGNAL_NOT_STABLE")

