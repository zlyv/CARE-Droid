#!/usr/bin/env python3
import json
from audit_common import *

def main():
    d=json.loads((OUTPUT/"collision_audit.json").read_text(encoding="utf-8")); replay=json.loads((OUTPUT/"replay_validation.json").read_text(encoding="utf-8"))
    lines=["# SVM Direct-C43 frozen representation collision audit v1","",f"Final status: **{d['status']}**.","",
      "This source-only audit replays the existing Direct-C43 protocol only to export hidden states. It does not train a new objective, tune a threshold, use a gate, or load future data.","",
      "## Frozen replay","",f"- Direct-C43 replay: {replay['status']}; max probability difference={replay['max_abs_probability_difference']:.8g}.",
      "- Frozen reference remains threshold=0.50, fixed=49, broken=17, net=32.","",
      "## Conditional information","",
      f"- Scalar FN-vs-TN AUC: {d['scalar_auc']['FN_VS_TN']:.6f}.",f"- Scalar TP-vs-FP AUC: {d['scalar_auc']['TP_VS_FP']:.6f}.",
      f"- Primary C43 probes passing both sides: {d['primary_passing_probes']}.","",
      "## Local collision","",f"- Hidden cosine k=10 purity (all routed): {d['hidden_cosine_purity_k10']:.6f}.",
      f"- Hidden cosine k=10 purity (broken): {d['broken_hidden_cosine_purity_k10']:.6f}.",
      f"- Matched-null broken-purity 95th percentile: {d['matched_null_broken_purity_95pct']:.6f}.","",
      "## Decision boundary","",
      "Only DIRECT_C43_READOUT_BOTTLENECK_SUPPORTED authorizes a separately specified readout experiment. C43_REPRESENTATION_COLLISION_SUPPORTED stops threshold, calibration, loss, and deeper-head tuning for this representation. An inconclusive result authorizes no model change.","",
      "Future rows loaded: **0**."]
    (OUTPUT/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    dump_json(OUTPUT/"final_decision.json",{"status":d["status"],"future_rows_loaded":0,"formal_integrity":"PENDING","next_action":"readout experiment only if readout bottleneck is supported; otherwise stop C43 decision tuning"})
    print("CARE_C43_REPRESENTATION_COLLISION_FINAL_STATUS="+d["status"],flush=True)
if __name__=="__main__":main()
