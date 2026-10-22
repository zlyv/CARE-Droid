#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from audit_common import *
from interventions import ARMS, intervention_semantics

def side_auc(y,base,route,score):
    benign=route&(base<.5); malware=route&(base>=.5)
    return {"FN_vs_TN_AUC":auc(y[benign],score[benign]),"FN":int(np.sum(benign&(y==1))),"TN":int(np.sum(benign&(y==0))),
      "TP_vs_FP_AUC":auc(y[malware],score[malware]),"TP":int(np.sum(malware&(y==1))),"FP":int(np.sum(malware&(y==0)))}

def main():
    saved=np.load(OUTPUT/"intervention_probabilities.npz",allow_pickle=False)
    ids,y,fold,base,route,_,_,_=load_reference(False)
    scores={arm:np.asarray(saved[arm+"__mean"],float) for arm in ARMS}
    final={arm:np.where(route,scores[arm],base) for arm in ARMS}
    full=metrics(y,base,final["FULL"],route); before=(base>=.5)==y; full_correct=(final["FULL"]>=.5)==y
    full_fixed=(~before)&full_correct; full_broken=before&(~full_correct)
    summary=[]; fold_rows=[]
    full_side=side_auc(y,base,route,scores["FULL"])
    for arm in ARMS:
        m=metrics(y,base,final[arm],route); s=side_auc(y,base,route,scores[arm]); correct=(final[arm]>=.5)==y
        row={"arm":arm,**m,**s,"delta_BA_vs_FULL":m["BA"]-full["BA"],"delta_net_vs_FULL":m["net"]-full["net"],
          "lost_FULL_fixed":int(np.sum(full_fixed&(~correct))),"rescued_FULL_broken":int(np.sum(full_broken&correct)),
          "prediction_changes_vs_FULL":int(np.sum((final[arm]>=.5)!=(final["FULL"]>=.5))),
          "FN_AUC_drop_vs_FULL":full_side["FN_vs_TN_AUC"]-s["FN_vs_TN_AUC"],
          "TP_AUC_drop_vs_FULL":full_side["TP_vs_FP_AUC"]-s["TP_vs_FP_AUC"]}
        summary.append(row)
        for f in FOLDS:
            sel=fold==f; fm=metrics(y[sel],base[sel],final[arm][sel],route[sel]); fs=side_auc(y[sel],base[sel],route[sel],scores[arm][sel])
            fold_rows.append({"arm":arm,"fold":f,**fm,**fs})
    write_csv(OUTPUT/"intervention_summary.csv",summary); write_csv(OUTPUT/"intervention_per_fold.csv",fold_rows)
    by={r["arm"]:r for r in summary}; fby={(r["arm"],r["fold"]):r for r in fold_rows}
    findings=[]
    for arm in ARMS[2:]:
        r=by[arm]; evaluable_fn=0; lower_fn=0; evaluable_tp=0; lower_tp=0
        for f in FOLDS:
            a=fby[(arm,f)]; q=fby[("FULL",f)]
            if np.isfinite(a["FN_vs_TN_AUC"]) and np.isfinite(q["FN_vs_TN_AUC"]): evaluable_fn+=1; lower_fn+=a["FN_vs_TN_AUC"]<q["FN_vs_TN_AUC"]
            if np.isfinite(a["TP_vs_FP_AUC"]) and np.isfinite(q["TP_vs_FP_AUC"]): evaluable_tp+=1; lower_tp+=a["TP_vs_FP_AUC"]<q["TP_vs_FP_AUC"]
        reliable=(r["FN_AUC_drop_vs_FULL"]>=.03 and r["TP_AUC_drop_vs_FULL"]>=.03 and lower_fn>=3 and lower_tp>=3)
        harmful=(r["rescued_FULL_broken"]>=4 and r["lost_FULL_fixed"]<=2 and r["delta_net_vs_FULL"]>=2)
        findings.append({"arm":arm,"reliable_both_sides":bool(reliable),"harmful_relation_candidate":bool(harmful),
          "folds_FN_AUC_lower":lower_fn,"folds_FN_evaluable":evaluable_fn,"folds_TP_AUC_lower":lower_tp,"folds_TP_evaluable":evaluable_tp,
          "FN_AUC_drop":r["FN_AUC_drop_vs_FULL"],"TP_AUC_drop":r["TP_AUC_drop_vs_FULL"],
          "lost_FULL_fixed":r["lost_FULL_fixed"],"rescued_FULL_broken":r["rescued_FULL_broken"],"delta_net":r["delta_net_vs_FULL"]})
    reliable=[r["arm"] for r in findings if r["reliable_both_sides"]]; harmful=[r["arm"] for r in findings if r["harmful_relation_candidate"]]
    if reliable: status="JOINT_DIRECTION_CONTRIBUTION_LOCALIZED"
    elif harmful: status="HARMFUL_ATTRIBUTION_RELATION_LOCALIZED"
    else: status="JOINT_CONTRIBUTION_INCONCLUSIVE"
    donor=read_csv(OUTPUT/"matched_donors.csv")
    donor_audit={"rows":len(donor),"same_stage1_direction":all(bool(int(r["stage1_direction"]) in (0,1)) for r in donor),
      "mean_base_abs_difference":float(np.mean([float(r["base_abs_difference"]) for r in donor])),
      "p95_base_abs_difference":float(np.quantile([float(r["base_abs_difference"]) for r in donor],.95)),
      "mean_log_host_abs_difference":float(np.mean([float(r["log_host_abs_difference"]) for r in donor]))}
    dump_json(OUTPUT/"joint_findings.json",{"status":status,"criteria_are_diagnostic_not_model_selection":True,
      "reliable_relations":reliable,"harmful_relation_candidates":harmful,"findings":findings,"donor_audit":donor_audit,"future_rows_loaded":0})
    sample=[]
    for i,sid in enumerate(ids):
        row={"sha256":sid,"fold":int(fold[i]),"y":int(y[i]),"stage1_probability":float(base[i]),"routed":int(route[i]),
          "stage1_correct":int(before[i]),"FULL_correct":int(full_correct[i]),"FULL_fixed":int(full_fixed[i]),"FULL_broken":int(full_broken[i])}
        for arm in ARMS: row[arm+"__probability"]=float(scores[arm][i]); row[arm+"__std"]=float(saved[arm+"__std"][i])
        sample.append(row)
    write_csv(OUTPUT/"sample_interventions.csv",sample)
    dump_json(OUTPUT/"final_decision.json",{"status":status,"frozen_reference":full,
      "reliable_relations":reliable,"harmful_relation_candidates":harmful,
      "next_action":"Only a separately frozen source ablation is authorized if a relation is localized; otherwise stop C43 component tuning.",
      "new_model_trained":False,"future_rows_loaded":0})
    semantics=intervention_semantics(); lines=["# SVM Direct-C43 joint contribution audit v1","",f"Final status: **{status}**.","",
      "The exact Direct-C43 model is trained once per frozen fold/seed. Every arm below is an inference-time, label-free correspondence intervention on that same model; no arm has extra parameters or a separately trained readout.","",
      "## Historical coverage","","Earlier experiments already tested modules marginally. This audit fills the missing conditional question: what happens when cross-block correspondence is destroyed while block marginals, route, threshold and model weights remain fixed.","",
      "## Pooled results","","| Arm | FN-vs-TN AUC | TP-vs-FP AUC | fixed | broken | net | lost full fixed | rescued full broken |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary: lines.append(f"| {r['arm']} | {r['FN_vs_TN_AUC']:.6f} | {r['TP_vs_FP_AUC']:.6f} | {r['fixed']} | {r['broken']} | {r['net']} | {r['lost_FULL_fixed']} | {r['rescued_FULL_broken']} |")
    lines += ["","## Interpretation boundary","","- A drop after a pairing intervention establishes model reliance on that correspondence; it does not prove causal Android program semantics.",
      "- Within-APK interventions preserve Host count and each affected block's marginal values. Matched APK swaps preserve donor marginals and Stage-I prediction direction without labels.",
      "- N0_ROW_ORDER must reproduce FULL and is the implementation negative control.",
      "- No future or external row was loaded. These results cannot select a temporal model.","",f"Reliable relations: {reliable or 'none'}.",f"Harmful-attribution candidates: {harmful or 'none'}.","", "Formal asset integrity is reported separately."]
    (OUTPUT/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("DIRECT_C43_JOINT_AUDIT_STATUS="+status,flush=True)
if __name__=="__main__": main()

