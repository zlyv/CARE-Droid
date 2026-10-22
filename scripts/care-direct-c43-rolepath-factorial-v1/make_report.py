#!/usr/bin/env python3
from ablation_common import *

def main():
    d=read_json(OUTPUT/"final_decision.json"); rows=read_csv(OUTPUT/"source_results.csv")
    lines=["# Direct-C43 RoleGraph query/readout factorial v1","",f"Final status: **{d['status']}**.","",
      "This is a strict 2017 source OOF attribution experiment. All four arms retain the same parameters and training protocol; only fixed RoleGraph query/readout path switches differ. No future row is loaded.","",
      "## Source results","","| Arm | BA | FN-vs-TN AUC | TP-vs-FP AUC | fixed | broken | net |","|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows: lines.append(f"| {r['arm']} | {float(r['BA']):.6f} | {float(r['FN_vs_TN_AUC']):.6f} | {float(r['TP_vs_FP_AUC']):.6f} | {r['fixed']} | {r['broken']} | {r['net']} |")
    q=d["factorial_effects"]["query"]; rr=d["factorial_effects"]["readout"]
    lines += ["","## Factorial attribution","",f"- Query path: FN effect={q['FN_effect']:.6f}, TP effect={q['TP_effect']:.6f}, both-positive folds={q['both_positive_folds']}/5, supported={q['supported']}.",
      f"- Readout path: FN effect={rr['FN_effect']:.6f}, TP effect={rr['TP_effect']:.6f}, both-positive folds={rr['both_positive_folds']}/5, supported={rr['supported']}.",
      f"- Simplification candidates: {d['simplification_candidates'] or 'none'}.","","## Boundary","",
      "The experiment identifies predictive contribution under fixed source protocol. It does not prove causal Android semantics and does not use future data to select an arm.","",f"Next action: {d['next_action']}."]
    (OUTPUT/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

if __name__=="__main__":main()
