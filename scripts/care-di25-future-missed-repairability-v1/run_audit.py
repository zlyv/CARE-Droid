#!/usr/bin/env python3
"""Counterfactual repairability of frozen DI25 misses using saved Q probabilities."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from audit_core import load_and_validate,write

OUT=Path("artifacts/care-di25-future-missed-repairability-v1")

def div(a,b):return float(a/b) if b else 0.0

def main():
    ids,years,y,base,route,expert,audit=load_and_validate()
    base_ok=(base>=.5)==y;expert_ok=(expert>=.5)==y;error=~base_ok
    summaries=[];samples=[]
    for year in (2020,2021):
        scope=years==year;caught=scope&error&route;missed=scope&error&~route
        missed_fix=missed&expert_ok;missed_unfixed=missed&~expert_ok
        unrouted_correct=scope&base_ok&~route;counterfactual_broken=unrouted_correct&~expert_ok
        for i in np.flatnonzero(missed):
            samples.append({"sha256":ids[i],"year":year,"label":int(y[i]),
                "error_type":"FN" if y[i]==1 else "FP","p_stage1":float(base[i]),
                "stage1_confidence":float(max(base[i],1-base[i])),"route_DI":0,
                "p_expert_Q":float(expert[i]),"Q_counterfactual_correct":int(expert_ok[i]),
                "counterfactual_status":"MISSED_FIXED_BY_Q" if expert_ok[i] else "MISSED_UNFIXED_BY_Q"})
        summaries.append({"scope":str(year),"n":int(scope.sum()),"stage1_errors":int((scope&error).sum()),
            "DI_caught_errors":int(caught.sum()),"DI_missed_errors":int(missed.sum()),
            "missed_repairable_by_Q":int(missed_fix.sum()),"missed_unrepairable_by_Q":int(missed_unfixed.sum()),
            "missed_repairability_rate":div(int(missed_fix.sum()),int(missed.sum())),
            "missed_FN":int((missed&(y==1)).sum()),"missed_FN_repaired":int((missed_fix&(y==1)).sum()),
            "missed_FP":int((missed&(y==0)).sum()),"missed_FP_repaired":int((missed_fix&(y==0)).sum()),
            "missed_highconf":int((missed&(np.maximum(base,1-base)>=.8)).sum()),
            "missed_highconf_repaired":int((missed_fix&(np.maximum(base,1-base)>=.8)).sum()),
            "caught_errors_repairable_by_Q":int((caught&expert_ok).sum()),
            "caught_error_repairability_rate":div(int((caught&expert_ok).sum()),int(caught.sum())),
            "unrouted_stage1_correct":int(unrouted_correct.sum()),
            "unrouted_correct_that_Q_would_break":int(counterfactual_broken.sum()),
            "force_all_unrouted_fixed":int(missed_fix.sum()),"force_all_unrouted_broken":int(counterfactual_broken.sum()),
            "force_all_unrouted_net":int(missed_fix.sum()-counterfactual_broken.sum())})
    scope=np.isin(years,(2020,2021));caught=scope&error&route;missed=scope&error&~route
    missed_fix=missed&expert_ok;unrouted_correct=scope&base_ok&~route;broken=unrouted_correct&~expert_ok
    summaries.append({"scope":"pooled_2020_2021","n":int(scope.sum()),"stage1_errors":int((scope&error).sum()),
        "DI_caught_errors":int(caught.sum()),"DI_missed_errors":int(missed.sum()),
        "missed_repairable_by_Q":int(missed_fix.sum()),"missed_unrepairable_by_Q":int((missed&~expert_ok).sum()),
        "missed_repairability_rate":div(int(missed_fix.sum()),int(missed.sum())),
        "missed_FN":int((missed&(y==1)).sum()),"missed_FN_repaired":int((missed_fix&(y==1)).sum()),
        "missed_FP":int((missed&(y==0)).sum()),"missed_FP_repaired":int((missed_fix&(y==0)).sum()),
        "missed_highconf":int((missed&(np.maximum(base,1-base)>=.8)).sum()),
        "missed_highconf_repaired":int((missed_fix&(np.maximum(base,1-base)>=.8)).sum()),
        "caught_errors_repairable_by_Q":int((caught&expert_ok).sum()),
        "caught_error_repairability_rate":div(int((caught&expert_ok).sum()),int(caught.sum())),
        "unrouted_stage1_correct":int(unrouted_correct.sum()),"unrouted_correct_that_Q_would_break":int(broken.sum()),
        "force_all_unrouted_fixed":int(missed_fix.sum()),"force_all_unrouted_broken":int(broken.sum()),
        "force_all_unrouted_net":int(missed_fix.sum()-broken.sum())})
    OUT.mkdir(parents=True,exist_ok=True);write(OUT/"summary.csv",summaries);write(OUT/"missed_error_counterfactuals.csv",samples)
    lines=["CARE-Droid frozen DI25-missed Q-Query repairability audit v1","",
      "Read-only post-hoc counterfactual using saved 2017-trained Q_QUERY_ONLY bounded probabilities.",
      "No training, inference, route change, threshold search, or future-driven method selection.",
      "Q was fit without applying the Router, so p_expert exists for routed and unrouted APKs.",""]
    for r in summaries:
        lines.append(f"{r['scope']}: missed={r['DI_missed_errors']} repairable={r['missed_repairable_by_Q']} "
          f"repairability={r['missed_repairability_rate']:.6f} FN={r['missed_FN']} FN_repaired={r['missed_FN_repaired']} "
          f"FP={r['missed_FP']} FP_repaired={r['missed_FP_repaired']} highconf={r['missed_highconf']} "
          f"highconf_repaired={r['missed_highconf_repaired']} caught_repairability={r['caught_error_repairability_rate']:.6f} "
          f"force_all_unrouted_fixed={r['force_all_unrouted_fixed']} broken={r['force_all_unrouted_broken']} net={r['force_all_unrouted_net']}")
    pooled=summaries[-1]
    if pooled["missed_repairability_rate"]<.25:decision="STAGE2_LIMITED_ON_ROUTER_MISSES"
    elif pooled["missed_repairability_rate"]>=.5:decision="ROUTER_BLIND_SPOT_WITH_SUBSTANTIAL_STAGE2_VALUE"
    else:decision="MIXED_ROUTER_AND_STAGE2_BOTTLENECK"
    lines += ["","DIAGNOSTIC_DECISION="+decision,
      "The force-all-unrouted line is an oracle/post-hoc safety diagnostic, not a deployable policy.",
      "2018-2021 are already-inspected diagnostic years and cannot authorize method revision."]
    (OUT/"REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("DI25_FUTURE_MISSED_REPAIRABILITY_AUDIT_DONE");print("\n".join(lines))

if __name__=="__main__":main()

