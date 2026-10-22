#!/usr/bin/env python3
from __future__ import annotations
import json
import numpy as np
from common import ROOT,read_csv,dump
repro=json.loads((ROOT/"formal_reproduction_audit.json").read_text());arms=read_csv(ROOT/"formal_ablation_results.csv");routing=read_csv(ROOT/"routing_analysis.csv");budget=read_csv(ROOT/"budget_curve.csv");cost=read_csv(ROOT/"cost_summary.csv");cost_status=json.loads((ROOT/"cost_analysis_status.json").read_text());internal=json.loads((ROOT/"internal_ablation_audit.json").read_text());bounded=json.loads((ROOT/"bounded_ablation_audit.json").read_text())
if repro["status"]!="PASS":raise RuntimeError("FORMAL_REPRODUCTION_FAILED")
def one(name):return next(r for r in arms if r["arm"]==name and r["aggregate_type"]!="std_across_5_frozen_seeds")
a0=one("A0_STAGE1_ONLY");a1=one("A1_RANDOM25");a2=one("A2_DI25_FORMAL");a3=one("A3_FULL100")
def pooled_route(kind):
    rows=[r for r in routing if r["router"]==kind]
    if kind=="DI25":groups={"FORMAL":rows}
    else:
        groups={seed:[r for r in rows if r["seed"]==seed] for seed in sorted(set(r["seed"] for r in rows))}
    vals=[]
    for seed,z in groups.items():
        total=sum(int(r["total_samples"]) for r in z);routed=sum(int(r["routed_samples"]) for r in z);errors=sum(int(r["stage1_errors"]) for r in z);re=sum(int(r["routed_stage1_errors"]) for r in z);ue=sum(int(r["unrouted_stage1_errors"]) for r in z)
        vals.append({"seed":seed,"error_recall":re/max(errors,1),"routing_precision":re/max(routed,1),"accepted_risk":ue/max(total-routed,1),"error_enrichment":re/max(routed,1)/max(errors/total,1e-12)})
    return {k:float(np.mean([v[k] for v in vals])) for k in ("error_recall","routing_precision","accepted_risk","error_enrichment")}
di=pooled_route("DI25");rr=pooled_route("RANDOM25")
formal_budget=next(r for r in budget if int(r["formal_budget"])==1)
summary={
 "CARE_DROID_FORMAL_V1_FROZEN":True,"formal_reproduction":"PASS","formal_method":"P31-RF -> RF-anchored MeanOpposition DI25 Top25% -> Role C43-RQ -> bounded correction",
 "A0_stage1_only":a0,"A1_random25_mean":a1,"A2_DI25_formal":a2,"A3_full100":a3,"A4_prebound":bounded,
 "DI25_vs_random25":{"DI25":di,"Random25_mean":rr,"error_recall_improvement":di["error_recall"]-rr["error_recall"],"accepted_risk_reduction":rr["accepted_risk"]-di["accepted_risk"],"BA_improvement":float(a2["BA"])-float(a1["BA"]),"net_improvement":float(a2["net"])-float(a1["net"])},
 "selective25_vs_full100":{"BA_difference_selective_minus_full":float(a2["BA"])-float(a3["BA"]),"net_difference_selective_minus_full":float(a2["net"])-float(a3["net"]),"broken_difference_selective_minus_full":float(a2["broken"])-float(a3["broken"]),"deep_analysis_count_reduction":int(float(a3["deep_analysis_count"])-float(a2["deep_analysis_count"])),"deep_analysis_work_reduction_fraction":1-200/799,"actual_runtime_reduction":None,"actual_runtime_claim_authorized":False},
 "formal_fixed_broken_reconstructed":{"fixed":repro["observed"]["fixed"],"broken":repro["observed"]["broken"],"net":repro["observed"]["net"]},
 "internal_C43_ablation":internal,"budget_curve":{"available":True,"formal_budget_percent":25,"used_for_selection":False,"formal_budget_row":formal_budget,"interpretation":"post-hoc characterization only; cannot change the frozen 25% budget"},
 "cost_analysis":cost_status,"cost_summary":cost,"future_rows_loaded":0,"selection_uses_future":False
}
dump(ROOT/"final_summary.json",summary)
table=[]
for row in (a0,a1,a2,a3):table.append({"Arm":row["arm"],"BA":float(row["BA"]),"AUC":float(row["AUC"]),"F1":float(row["F1"]),"fixed":float(row["fixed"]),"broken":float(row["broken"]),"net":float(row["net"]),"deep_analysis_percent":100*float(row["deep_analysis_ratio"]),"relative_cost":"UNAVAILABLE"})
lines=["CARE-Droid Formal v1 freeze and ablation summary","",f"FORMAL_REPRODUCTION=PASS max_prediction_diff={repro['prediction_reconstruction_max_abs_diff']}","Arm,BA,AUC,F1,fixed,broken,net,deep-analysis %,relative cost"]
for r in table:lines.append(f"{r['Arm']},{r['BA']:.6f},{r['AUC']:.6f},{r['F1']:.6f},{r['fixed']:.3f},{r['broken']:.3f},{r['net']:.3f},{r['deep_analysis_percent']:.3f},{r['relative_cost']}")
lines += ["",f"DI25_error_recall={di['error_recall']:.6f} Random25_error_recall={rr['error_recall']:.6f} delta={di['error_recall']-rr['error_recall']:+.6f}",f"DI25_accepted_risk={di['accepted_risk']:.6f} Random25_accepted_risk={rr['accepted_risk']:.6f} reduction={rr['accepted_risk']-di['accepted_risk']:+.6f}",f"A4={bounded['status']}","BUDGET_CURVE_USED_FOR_SELECTION=false",f"COST_STATUS={cost_status['status']} actual runtime savings are not claimed",f"INTERNAL_ABLATIONS_COMPATIBLE={internal['compatible_formal_internal_ablations']}","future_rows_loaded = 0","CARE_DROID_FORMAL_V1_FROZEN = true"]
(ROOT/"REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8");print("\n".join(lines))

