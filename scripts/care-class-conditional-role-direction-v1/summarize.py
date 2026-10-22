#!/usr/bin/env python3
from __future__ import annotations
import math,json
import numpy as np
from common import ROOT,read_csv,write_csv,dump

agg=read_csv(ROOT/"source_aggregate.csv");fold=read_csv(ROOT/"source_fold_results.csv");support=read_csv(ROOT/"role_prototype_support.csv");sep=read_csv(ROOT/"prototype_separation_audit.csv")
A={r["arm"]:r for r in agg}
def f(a,k):return float(A[a][k])
def i(a,k):return int(float(A[a][k]))
d0="D0_CURRENT_C43"
baseline_ok=(abs(f(d0,"BA")-.9323840852130325)<1e-10 and i(d0,"fixed")==11 and i(d0,"broken")==7 and i(d0,"net")==4)
valid_by_fold={q:sum(int(float(r["prototype_valid"])) for r in support if int(float(r["fold"]))==q) for q in range(1,6)}
valid_sep=[float(r["fisher_like_separation"]) for r in sep if int(float(r["prototype_valid"])) and str(r["fisher_like_separation"]).lower()!="nan"]
collapse_fraction=float(np.mean([float(r["prototype_cosine"])>=.98 for r in support if int(float(r["prototype_valid"]))]))
fold_summary=[]
for arm in ("D0_CURRENT_C43","D1_GLOBAL","D2_PER_ROLE","D3_STAGE_CONDITIONED","C1_CLASS_SHUFFLE","C2_ROLE_SHUFFLE"):
    z=[r for r in fold if r["arm"]==arm]
    row={"arm":arm}
    for k in ("BA","AUC","F1","net"):
        v=np.asarray([float(r[k]) for r in z]);row[k+"_fold_mean"]=float(v.mean());row[k+"_fold_std"]=float(v.std())
    fold_summary.append(row)
write_csv(ROOT/"source_fold_summary.csv",fold_summary)
FS={r["arm"]:r for r in fold_summary}

def noninferior_folds(arm):
    out=0
    for q in range(1,6):
        z=next(r for r in fold if r["arm"]==arm and int(float(r["fold"]))==q);b=next(r for r in fold if r["arm"]==d0 and int(float(r["fold"]))==q)
        out+=int(int(float(z["net"]))>=int(float(b["net"])))
    return out

# Predeclared, deliberately multi-criterion source rule.  D2/D3 must improve
# correction utility and both direction sides; a null-control gap is required.
def passed(arm):
    return (
      baseline_ok and i(arm,"net")>i(d0,"net") and i(arm,"broken")<=i(d0,"broken")
      and f(arm,"FP_vs_TP_AUC")>=f(d0,"FP_vs_TP_AUC")+.05
      and f(arm,"FN_vs_TN_AUC")>=f(d0,"FN_vs_TN_AUC")-.03
      and f(arm,"FIXED_vs_BROKEN_AUC")>=.55 and noninferior_folds(arm)>=3
      and f("D2_PER_ROLE","FP_vs_TP_AUC")>=f("C1_CLASS_SHUFFLE","FP_vs_TP_AUC")+.05
      and f("D2_PER_ROLE","FIXED_vs_BROKEN_AUC")>=f("C1_CLASS_SHUFFLE","FIXED_vs_BROKEN_AUC")+.05
      and f("D2_PER_ROLE","FP_vs_TP_AUC")>=f("C2_ROLE_SHUFFLE","FP_vs_TP_AUC")+.03
      and f("D2_PER_ROLE","FP_vs_TP_AUC")>f("D1_GLOBAL","FP_vs_TP_AUC")
    )
candidates=[a for a in ("D2_PER_ROLE","D3_STAGE_CONDITIONED") if passed(a)]
selected=max(candidates,key=lambda a:(i(a,"net"),-i(a,"broken"),f(a,"BA"))) if candidates else None
status="CLASS_CONDITIONAL_ROLE_DIRECTION_SUPPORTED" if selected else "CLASS_CONDITIONAL_ROLE_DIRECTION_NOT_SUPPORTED"
summary={
 "status":status,"selected_arm":selected,"baseline_reproduced":baseline_ok,
 "existing_experiment_status":json.loads((ROOT/"existing_experiment_audit.json").read_text())["status"],
 "source_rows":799,"route":200,"future_rows_loaded":0,"selection_uses_future":False,
 "aggregate":agg,"valid_prototypes_per_fold":valid_by_fold,"prototype_collapse_fraction_cosine_ge_0_98":collapse_fraction,
 "fold_summary":fold_summary,
 "mean_valid_role_fisher_separation":float(np.mean(valid_sep)) if valid_sep else None,
 "noninferior_net_folds":{"D1_GLOBAL":noninferior_folds("D1_GLOBAL"),"D2_PER_ROLE":noninferior_folds("D2_PER_ROLE"),"D3_STAGE_CONDITIONED":noninferior_folds("D3_STAGE_CONDITIONED")},
 "support_rule":"D2/D3 net>D0; broken<=D0; FP-vs-TP gain>=.05; FN-vs-TN loss<=.03; FIXED-vs-BROKEN AUC>=.55; >=3/5 noninferior-net folds; D2 FP and FIXED/BROKEN AUC each >= C1+0.05; D2 FP>=C2+0.03; D2 FP AUC>D1",
 "D4_executed":False,"D4_reason":"D4 is not needed for the primary D0-D3 sufficiency decision; no larger model is used to rescue a weak D2.",
 "interpretation":"A positive direction score always means malware support; FP-vs-TP uses its negation as benign opposition."
}
dump(ROOT/"final_summary.json",summary)
lines=["CARE-Droid Class-Conditional Role Direction v1","","Strict 2017 source-only; P31-RF / RF-DI25 / A0 Role C43-RQ fixed; future_rows_loaded=0.",f"Existing experiment audit: {summary['existing_experiment_status']}"]
for arm in (d0,"D1_GLOBAL","D2_PER_ROLE","D3_STAGE_CONDITIONED"):
    lines.append(f"{arm}: BA={f(arm,'BA'):.6f} AUC={f(arm,'AUC'):.6f} F1={f(arm,'F1'):.6f} fixed={i(arm,'fixed')} broken={i(arm,'broken')} net={i(arm,'net')} direction={f(arm,'routed_direction_accuracy'):.6f} FP_vs_TP_AUC={f(arm,'FP_vs_TP_AUC'):.6f} FN_vs_TN_AUC={f(arm,'FN_vs_TN_AUC'):.6f} FIXED_vs_BROKEN_AUC={f(arm,'FIXED_vs_BROKEN_AUC'):.6f} BA_fold={float(FS[arm]['BA_fold_mean']):.6f}+/-{float(FS[arm]['BA_fold_std']):.6f} noninferior_folds={noninferior_folds(arm)}/5")
lines += ["",f"CLASS_SHUFFLE_RESULT: D2_FP_AUC_minus_C1={f('D2_PER_ROLE','FP_vs_TP_AUC')-f('C1_CLASS_SHUFFLE','FP_vs_TP_AUC'):+.6f} D2_FIXBROKEN_AUC_minus_C1={f('D2_PER_ROLE','FIXED_vs_BROKEN_AUC')-f('C1_CLASS_SHUFFLE','FIXED_vs_BROKEN_AUC'):+.6f}",f"ROLE_SHUFFLE_RESULT: D2_FP_AUC_minus_C2={f('D2_PER_ROLE','FP_vs_TP_AUC')-f('C2_ROLE_SHUFFLE','FP_vs_TP_AUC'):+.6f}",f"PROTOTYPE_SUPPORT: valid_roles_per_fold={valid_by_fold} collapse_fraction={collapse_fraction:.6f}","future_rows_loaded=0",status]
(ROOT/"REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
print("\n".join(lines))
