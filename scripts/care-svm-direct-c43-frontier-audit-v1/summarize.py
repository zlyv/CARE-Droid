#!/usr/bin/env python3
from common import OUTPUT,read_json
def main():
    d=read_json(OUTPUT/"final_decision.json");i=read_json(OUTPUT/"baseline_integrity_check.json");b=d["best_net_descriptive"];m=d["minimum_broken_with_fixed_at_least_40"]
    lines=["# SVM Direct-C43 Unified Decision Frontier Audit v1","",f"Final status: **{d['status']}**.","","This is retrospective source anatomy, not threshold selection.","Only one global threshold is evaluated; no prediction-side rule, class-specific threshold, gate, future data or model retraining is used.","","## Frozen references","","- Bounded C43: fixed=29, broken=4, net=25.","- Direct C43 at 0.50: fixed=49, broken=17, net=32.","","## Frontier","",f"- Descriptive maximum net: threshold={b['threshold']:.2f}, fixed={b['fixed']}, broken={b['broken']}, net={b['net']}."]
    if m:lines.append(f"- Minimum broken while retaining at least 40 fixed: threshold={m['threshold']:.2f}, fixed={m['fixed']}, broken={m['broken']}, net={m['net']}.")
    lines += [f"- Strict Pareto regions: {d['strict_pareto_regions']}",f"- Broad practical regions: {d['broad_practical_regions']}","","## Decision boundary",""]
    if d["status"]=="DIRECT_C43_STRICT_PARETO_REGION_FOUND":lines.append("A global region Pareto-dominates the bounded reference. Authorize a separate strict nested calibration experiment; do not deploy a threshold from this audit.")
    elif d["status"]=="DIRECT_C43_BROAD_FRONTIER_WORTH_NESTED_TEST":lines.append("A broad practical region exists. A separate strict nested calibration test is justified, but this audit cannot select its threshold.")
    else:lines.append("No stable global threshold region satisfies the frozen trade-off. Stop threshold tuning; score overlap rather than a simple offset is the remaining limitation.")
    lines += ["",f"Future rows loaded: {d['future_rows_loaded']}.",f"Formal asset integrity: {i['status']}."]
    (OUTPUT/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print("CARE_DIRECT_C43_FRONTIER_FINAL_STATUS="+d["status"],flush=True)
if __name__=="__main__":main()
