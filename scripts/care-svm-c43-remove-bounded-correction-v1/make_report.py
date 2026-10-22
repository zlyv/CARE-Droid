#!/usr/bin/env python3
import csv
from ablation_common import OUTPUT, ensure_output, read_json


def read(path):
    with path.open("r",encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))


def main():
    decision=read_json(OUTPUT/"final_decision.json"); results=read(OUTPUT/"source_results.csv"); paired=read(OUTPUT/"paired_vs_bounded.csv"); raw=read(OUTPUT/"raw_output_audit.csv")
    lines=["# CARE-Droid SVM+C43 removal of bounded correction v1","","Strict 2017 fixed-fold OOF; future rows loaded: 0.","",
           "| Arm | BA | AUC | F1 | fixed | broken | net |","|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:lines.append(f"| {r['arm']} | {float(r['BA']):.6f} | {float(r['AUC']):.6f} | {float(r['F1']):.6f} | {r['fixed']} | {r['broken']} | {r['net']} |")
    lines += ["","## Paired comparison with formal bounded C43","","| Arm | BA gain pp | candidate-only correct | bounded-only correct | p | non-inferior folds |","|---|---:|---:|---:|---:|---:|"]
    for r in paired:lines.append(f"| {r['arm']} | {float(r['BA_gain_pp_vs_bounded']):+.4f} | {r['candidate_only_correct']} | {r['bounded_only_correct']} | {float(r['mcnemar_exact_p']):.6f} | {r['noninferior_folds']}/5 |")
    lines += ["","## Raw-output stability","","| Arm | abs logit p50 | p95 | max | >10 |","|---|---:|---:|---:|---:|"]
    for r in raw:lines.append(f"| {r['arm']} | {float(r['abs_logit_p50']):.4f} | {float(r['abs_logit_p95']):.4f} | {float(r['abs_logit_max']):.4f} | {r['abs_logit_gt_10']} |")
    lines += ["","## Decision","",f"**{decision['status']}**","",f"Supported arms: {decision['supported_arms']}",f"Baseline integrity: {decision['baseline_integrity']}",""]
    ensure_output(OUTPUT/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    print("SVM_C43_REMOVE_BOUNDED_FINAL_STATUS="+decision["status"],flush=True);print("CARE_SVM_C43_REMOVE_BOUNDED_PIPELINE_DONE",flush=True)


if __name__=="__main__":main()
