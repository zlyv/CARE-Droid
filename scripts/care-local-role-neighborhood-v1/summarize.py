#!/usr/bin/env python3
from __future__ import annotations
import json
import numpy as np
from common import ROOT,read_csv,dump

decision=json.loads((ROOT/"probe_decision.json").read_text());probe=read_csv(ROOT/"direction_probe_results.csv");folds=read_csv(ROOT/"fold_direction_results.csv");near=read_csv(ROOT/"near_duplicate_audit.csv");size=read_csv(ROOT/"size_shortcut_audit.csv")
arms=("L0_CURRENT_C43","L1_APK_KNN","L2_ROLE_KNN","L3_STAGE1_CONDITIONED")
comparisons=("FP_vs_TP","FN_vs_TN","D0_FIXED_vs_D0_BROKEN","ROUTED_ERROR_vs_CORRECT","MALWARE_vs_BENIGN")
table={arm:{} for arm in arms}
for row in probe:
    if row["arm"] in table and row["comparison"] in comparisons:table[row["arm"]][row["comparison"]]={"AUC":None if row["AUC"]=="" else float(row["AUC"]),"status":row["status"],"positive_n":int(row["positive_n"]),"negative_n":int(row["negative_n"])}
fold_consistency=[]
for arm in arms:
    for comparison in ("FP_vs_TP","FN_vs_TN","D0_FIXED_vs_D0_BROKEN"):
        valid=[r for r in folds if r["arm"]==arm and r["comparison"]==comparison and r["status"]=="DEFINED"]
        fold_consistency.append({"arm":arm,"comparison":comparison,"supporting_folds":sum(float(r["AUC"])>.5 for r in valid),"valid_folds":len(valid),"fold_AUCs":[{"fold":int(r["fold"]),"AUC":float(r["AUC"])} for r in valid]})
def control_summary(path):
    rows=read_csv(path);out=[]
    for control in sorted(set(r["control"] for r in rows)):
        for comparison in sorted(set(r["comparison"] for r in rows)):
            vals=[float(r["AUC"]) for r in rows if r["control"]==control and r["comparison"]==comparison and r["status"]=="DEFINED"]
            if vals:out.append({"control":control,"comparison":comparison,"repetitions":len(vals),"mean_AUC":float(np.mean(vals)),"std_AUC":float(np.std(vals)),"q05":float(np.quantile(vals,.05)),"q95":float(np.quantile(vals,.95))})
    return out
controls={"label_shuffle":control_summary(ROOT/"label_shuffle_results.csv"),"random_neighbor":control_summary(ROOT/"random_neighbor_results.csv"),"role_shuffle":control_summary(ROOT/"role_shuffle_results.csv")}
near_summary=[r for r in near];max_size_corr={arm:max([abs(float(r["spearman"])) for r in size if r["arm"]==arm],default=None) for arm in ("L1_APK_KNN","L2_ROLE_KNN","L3_STAGE1_CONDITIONED")}
summary={"decision":decision["decision"],"selected_probe":decision["selected_probe"],"representation":"label-free Compact43 Host content grouped by SelfRole31 (not an unavailable learned hidden embedding)","k":5,"minimum_per_class_role_support":5,"direction_results":table,"fold_consistency":fold_consistency,"matched_controls":controls,"near_duplicate_audit":near_summary,"max_abs_size_spearman":max_size_corr,"advance_assessments":decision["assessments"],"integration_ran":decision["integration_ran"],"future_rows_loaded":0,"selection_uses_future":False}
if (ROOT/"integration_results.csv").exists():summary["integration_results"]=read_csv(ROOT/"integration_results.csv")
dump(ROOT/"final_summary.json",summary)
lines=["CARE-Droid Local Role Neighborhood Probe v1","","Strict 2017 source-only; P31-RF / RF-DI25 / A0 Role C43-RQ frozen; future_rows_loaded=0.","Representation: label-free Compact43 Host content grouped by SelfRole31; no learned hidden embedding is claimed."]
for arm in arms:
    m=table[arm];lines.append(f"{arm}: FP-vs-TP={m['FP_vs_TP']['AUC']} FN-vs-TN={m['FN_vs_TN']['AUC']} FIXED-vs-BROKEN={m['D0_FIXED_vs_D0_BROKEN']['AUC']} routed-error={m['ROUTED_ERROR_vs_CORRECT']['AUC']}")
lines += ["",f"VALID_FOLD_CONSISTENCY={json.dumps(fold_consistency,ensure_ascii=False)}",f"REAL_VS_LABEL_SHUFFLE={json.dumps(controls['label_shuffle'],ensure_ascii=False)}",f"REAL_VS_RANDOM_NEIGHBOR={json.dumps(controls['random_neighbor'],ensure_ascii=False)}",f"REAL_ROLE_VS_ROLE_SHUFFLE={json.dumps(controls['role_shuffle'],ensure_ascii=False)}",f"NEAR_DUPLICATE_SENSITIVITY={json.dumps([r for r in near_summary if r.get('record_type')=='exclude_ge_099_performance'],ensure_ascii=False)}",f"MAX_ABS_SIZE_SPEARMAN={json.dumps(max_size_corr)}","future_rows_loaded=0",decision["decision"]]
(ROOT/"REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
print("\n".join(lines))

