#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from audit_common import *
from interventions import ARMS, BASE_CALIPER, LOG_HOST_CALIPER, P31_HELLINGER_CALIPER

def side_auc(y, base, route, score, eligible=None):
    eligible=np.ones(len(y),bool) if eligible is None else np.asarray(eligible,bool)
    benign=route&eligible&(base<.5); malware=route&eligible&(base>=.5)
    return {"FN_vs_TN_AUC":auc(y[benign],score[benign]),
      "routed_predicted_benign":int(benign.sum()),"side_FN":int(np.sum(benign&(y==1))),"side_TN":int(np.sum(benign&(y==0))),
      "TP_vs_FP_AUC":auc(y[malware],score[malware]),
      "routed_predicted_malware":int(malware.sum()),"side_TP":int(np.sum(malware&(y==1))),"side_FP":int(np.sum(malware&(y==0)))}

def main():
    saved=np.load(OUTPUT/"intervention_probabilities.npz",allow_pickle=False)
    ids,y,fold,base,route,_,_,_=load_reference(False)
    eligible=np.asarray(saved["STRICT_MATCH__eligible"],bool)
    scores={arm:np.asarray(saved[arm+"__mean"],float) for arm in ARMS}
    final={arm:np.where(route,scores[arm],base) for arm in ARMS}
    full=metrics(y,base,final["FULL"],route); before=(base>=.5)==y; full_correct=(final["FULL"]>=.5)==y
    full_fixed=(~before)&full_correct; full_broken=before&(~full_correct); strict_route=route&eligible
    summary=[]; fold_rows=[]; full_side=side_auc(y,base,route,scores["FULL"],eligible)
    for arm in ARMS:
        m=metrics(y,base,final[arm],route); s=side_auc(y,base,route,scores[arm],eligible); correct=(final[arm]>=.5)==y
        row={"arm":arm,**m,**s,"strict_matched_routed":int(strict_route.sum()),
          "delta_BA_vs_FULL":m["BA"]-full["BA"],"delta_net_vs_FULL":m["net"]-full["net"],
          "lost_FULL_fixed":int(np.sum(full_fixed&(~correct))),"rescued_FULL_broken":int(np.sum(full_broken&correct)),
          "prediction_changes_vs_FULL":int(np.sum((final[arm]>=.5)!=(final["FULL"]>=.5))),
          "FN_AUC_drop_vs_FULL":full_side["FN_vs_TN_AUC"]-s["FN_vs_TN_AUC"],
          "TP_AUC_drop_vs_FULL":full_side["TP_vs_FP_AUC"]-s["TP_vs_FP_AUC"]}
        summary.append(row)
        for f in FOLDS:
            sel=fold==f; fm=metrics(y[sel],base[sel],final[arm][sel],route[sel])
            fs=side_auc(y[sel],base[sel],route[sel],scores[arm][sel],eligible[sel])
            fold_rows.append({"arm":arm,"fold":f,**fm,**fs,"strict_matched_routed":int(np.sum(strict_route&sel))})
    write_csv(OUTPUT/"intervention_summary.csv",summary); write_csv(OUTPUT/"intervention_per_fold.csv",fold_rows)
    by={r["arm"]:r for r in summary}; fby={(r["arm"],r["fold"]):r for r in fold_rows}
    coverage=float(np.sum(strict_route)/max(np.sum(route),1)); fold_coverage=[int(np.sum(strict_route&(fold==f))) for f in FOLDS]
    coverage_pass=coverage>=.75 and sum(v>=28 for v in fold_coverage)>=4
    findings=[]
    for arm in ARMS[2:]:
        r=by[arm]; ef=lf=et=lt=0
        for f in FOLDS:
            a=fby[(arm,f)]; q=fby[("FULL",f)]
            if np.isfinite(a["FN_vs_TN_AUC"]) and np.isfinite(q["FN_vs_TN_AUC"]): ef+=1; lf+=a["FN_vs_TN_AUC"]<q["FN_vs_TN_AUC"]
            if np.isfinite(a["TP_vs_FP_AUC"]) and np.isfinite(q["TP_vs_FP_AUC"]): et+=1; lt+=a["TP_vs_FP_AUC"]<q["TP_vs_FP_AUC"]
        reliable=bool(r["FN_AUC_drop_vs_FULL"]>=.03 and r["TP_AUC_drop_vs_FULL"]>=.03 and lf>=3 and lt>=3)
        findings.append({"arm":arm,"reliable_both_sides":reliable,"folds_FN_AUC_lower":lf,"folds_FN_evaluable":ef,
          "folds_TP_AUC_lower":lt,"folds_TP_evaluable":et,"FN_AUC_drop":r["FN_AUC_drop_vs_FULL"],
          "TP_AUC_drop":r["TP_AUC_drop_vs_FULL"],"lost_FULL_fixed":r["lost_FULL_fixed"],
          "rescued_FULL_broken":r["rescued_FULL_broken"],"delta_net":r["delta_net_vs_FULL"]})
    reliable=[r["arm"] for r in findings if r["reliable_both_sides"]]
    role_confirmed=all(x in reliable for x in ("I5_ROLEGRAPH_LOCAL_QUERY_ALIGNMENT","I6_ROLEGRAPH_WHOLE_ALIGNMENT"))
    if not coverage_pass: status="INSUFFICIENT_STRICT_MATCH_COVERAGE"
    elif role_confirmed: status="STRICT_MATCHED_ROLEGRAPH_CONTRIBUTION_CONFIRMED"
    else: status="ROLEGRAPH_CONTRIBUTION_NOT_CONFIRMED_UNDER_STRICT_MATCHING"
    donor=read_csv(OUTPUT/"matched_donors.csv"); matched=[r for r in donor if int(r.get("matched",0))==1]
    donor_audit={"rows":len(donor),"matched_rows":len(matched),"matched_unique_targets":len({r["sha256"] for r in matched}),
      "same_stage1_direction":all(int(r["stage1_direction"])==int(r["donor_stage1_direction"]) for r in matched),
      "mean_base_abs_difference":float(np.mean([float(r["base_abs_difference"]) for r in matched])) if matched else None,
      "p95_base_abs_difference":float(np.quantile([float(r["base_abs_difference"]) for r in matched],.95)) if matched else None,
      "mean_log_host_abs_difference":float(np.mean([float(r["log_host_abs_difference"]) for r in matched])) if matched else None,
      "p95_log_host_abs_difference":float(np.quantile([float(r["log_host_abs_difference"]) for r in matched],.95)) if matched else None,
      "mean_p31_hellinger_distance":float(np.mean([float(r["p31_hellinger_distance"]) for r in matched])) if matched else None,
      "p95_p31_hellinger_distance":float(np.quantile([float(r["p31_hellinger_distance"]) for r in matched],.95)) if matched else None,
      "calipers":{"base":BASE_CALIPER,"log_host":LOG_HOST_CALIPER,"p31_hellinger":P31_HELLINGER_CALIPER},
      "strict_routed_coverage":coverage,"strict_routed_by_fold":fold_coverage}
    dump_json(OUTPUT/"strict_matched_findings.json",{"status":status,"coverage_pass":coverage_pass,
      "rolegraph_confirmed":role_confirmed,"reliable_relations":reliable,"findings":findings,
      "donor_audit":donor_audit,"criteria_are_diagnostic_not_model_selection":True,"future_rows_loaded":0})
    sample=[]
    for i,sid in enumerate(ids):
        row={"sha256":sid,"fold":int(fold[i]),"y":int(y[i]),"stage1_probability":float(base[i]),
          "routed":int(route[i]),"strict_match_eligible":int(eligible[i]),"stage1_correct":int(before[i]),
          "FULL_correct":int(full_correct[i]),"FULL_fixed":int(full_fixed[i]),"FULL_broken":int(full_broken[i])}
        for arm in ARMS: row[arm+"__probability"]=float(scores[arm][i]); row[arm+"__std"]=float(saved[arm+"__std"][i])
        sample.append(row)
    write_csv(OUTPUT/"sample_interventions.csv",sample)
    next_action=("run a separately frozen query/readout factorial source ablation" if status=="STRICT_MATCHED_ROLEGRAPH_CONTRIBUTION_CONFIRMED" else "stop RoleGraph component tuning")
    dump_json(OUTPUT/"final_decision.json",{"status":status,"frozen_reference":full,"strict_matched_routed_coverage":coverage,
      "strict_matched_routed_by_fold":fold_coverage,"reliable_relations":reliable,"next_action":next_action,
      "new_model_trained":False,"future_rows_loaded":0})
    lines=["# SVM Direct-C43 strict matched joint audit v1","",f"Final status: **{status}**.","",
      "This source-only audit repeats only the previously localized RoleGraph/Global24 interventions with strict, label-free P31/base/Host calipers. Unmatched targets remain visible and are excluded from conditional AUC decisions.","",
      "## Match audit","",f"- Routed strict-match coverage: {coverage:.6f} ({int(strict_route.sum())}/200).",
      f"- Per-fold matched routed counts: {fold_coverage}.",f"- Calipers: base={BASE_CALIPER}, log Host={LOG_HOST_CALIPER}, P31 Hellinger={P31_HELLINGER_CALIPER}.","",
      "## Results","","| Arm | FN-vs-TN AUC | TP-vs-FP AUC | fixed | broken | net |","|---|---:|---:|---:|---:|---:|"]
    for r in summary: lines.append(f"| {r['arm']} | {r['FN_vs_TN_AUC']:.6f} | {r['TP_vs_FP_AUC']:.6f} | {r['fixed']} | {r['broken']} | {r['net']} |")
    lines += ["","## Boundary","","This establishes frozen-model reliance under a stricter matched counterfactual. It does not by itself prove causal Android semantics or authorize future testing.","",f"Next action: {next_action}."]
    (OUTPUT/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("DIRECT_C43_STRICT_MATCHED_STATUS="+status,flush=True)

if __name__=="__main__": main()
