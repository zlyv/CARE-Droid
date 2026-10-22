#!/usr/bin/env python3
from common import OUTPUT, read_json

def main():
    d=read_json(OUTPUT/"final_decision.json"); integrity=read_json(OUTPUT/"baseline_integrity_check.json")
    lines=["# CARE-Droid C43 Conditional Matched Ranking v1","",f"Final status: **{d['status']}**.","",
           "This source-only probe asks whether one unified C43 malware score can rank opposite-label APKs after controlling the frozen Stage-I view.",
           "It does not add a gate, prototype, Host pseudo-label, future selection, correction rule, or end-to-end prediction.","","## Routed source results",""]
    lines.append("| Arm | FN-vs-TN AUC | TP-vs-FP AUC | FN folds >0.5 | TP folds >0.5 |")
    lines.append("|---|---:|---:|---:|---:|")
    for arm,v in d["results"].items():
        lines.append(f"| {arm} | {v['FN_vs_TN']['auc']:.6f} | {v['TP_vs_FP']['auc']:.6f} | {v['folds_FN_above_random']} | {v['folds_TP_above_random']} |")
    lines += ["",f"Best non-permutation null: FN={d['best_null']['FN']:.6f}, TP={d['best_null']['TP']:.6f}.",
              f"Label-shuffle 95th percentile: FN={d['label_shuffle_95pct']['FN_vs_TN_AUC']:.6f}, TP={d['label_shuffle_95pct']['TP_vs_FP_AUC']:.6f}.","","## Boundary",""]
    if d["status"]=="CONDITIONAL_MATCHED_C43_SUPPORTED": lines.append("Both directional probes passed. A separate, newly frozen integration experiment is authorized; this run itself does not change predictions.")
    elif d["status"]=="CONDITIONAL_MATCHED_C43_PARTIALLY_SUPPORTED": lines.append("Only one direction is supported. Do not claim bidirectional correction and do not run end-to-end integration under this protocol.")
    else: lines.append("Conditional matching did not beat the controls on both sides. Freeze the negative result; do not rescue it with a larger model.")
    lines += ["",f"Future rows loaded: {d['future_rows_loaded']}.",f"Formal asset integrity: {integrity['status']}."]
    (OUTPUT/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("CARE_C43_CONDITIONAL_MATCHED_FINAL_STATUS="+d["status"],flush=True)
if __name__=="__main__": main()
