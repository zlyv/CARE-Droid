#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from ablation_common import ARMS, FORMAL, METHOD, OUTPUT, dump_json, formal_rows, mcnemar_exact, metrics, norm, read_csv, write_csv


def main():
    formal=formal_rows(); ids=[norm(r["sha256"]) for r in formal]; y=np.asarray([int(float(r["label"])) for r in formal]); fold=np.asarray([int(float(r["fold"])) for r in formal])
    base=np.asarray([float(r["stage1_prob"]) for r in formal]); bounded=np.asarray([float(r["stage2_prob"]) for r in formal]); route=np.asarray([int(float(r["routed"])) for r in formal],bool)
    arms={"S0_STAGE1_SVM":base,"B0_FORMAL_BOUNDED":bounded}; raw_audit=[]
    for arm in ARMS:
        method=METHOD[arm]; path=OUTPUT/"source"/arm/"source_predictions.csv"; rows=read_csv(path); by={norm(r["sha256"]):r for r in rows}
        if set(by)!=set(ids): raise RuntimeError("BLOCKED_BY_CONTRACT: SHA mismatch for "+arm)
        ordered=[by[sid] for sid in ids]; candidate=np.asarray([float(r[method+"__prob"]) for r in ordered]); candidate_route=np.asarray([int(float(r["A25__routed"])) for r in ordered],bool)
        candidate_base=np.asarray([float(r["Global24__prob"]) for r in ordered])
        if not np.array_equal(candidate_route,route) or not np.allclose(candidate_base,base,atol=1e-12): raise RuntimeError("BLOCKED_BY_CONTRACT: Stage-I/router changed for "+arm)
        if not np.all(np.isfinite(candidate)): raise RuntimeError("BLOCKED_BY_CONTRACT: non-finite probability for "+arm)
        arms[arm]=candidate
        expert=np.asarray([float(r[method+"__expert_prob"]) for r in ordered]); clipped=np.clip(expert,1e-12,1-1e-12); logits=np.log(clipped/(1-clipped))
        if arm.startswith("U1"): logits=logits-np.log(np.clip(base,1e-12,1-1e-12)/np.clip(1-base,1e-12,1-1e-12))
        selected=logits[route]; raw_audit.append({"arm":arm,"routed":int(route.sum()),"abs_logit_p50":float(np.median(np.abs(selected))),
            "abs_logit_p90":float(np.quantile(np.abs(selected),.9)),"abs_logit_p95":float(np.quantile(np.abs(selected),.95)),
            "abs_logit_max":float(np.max(np.abs(selected))),"abs_logit_gt_10":int(np.sum(np.abs(selected)>10))})
    pooled=[]; per_fold=[]
    for name,prob in arms.items():
        pooled.append({"arm":name,**metrics(y,base,prob,route)})
        for current in range(1,6):
            s=fold==current; per_fold.append({"arm":name,"fold":current,**metrics(y[s],base[s],prob[s],route[s])})
    write_csv(OUTPUT/"source_results.csv",pooled); write_csv(OUTPUT/"source_fold_results.csv",per_fold); write_csv(OUTPUT/"raw_output_audit.csv",raw_audit)
    baseline=arms["B0_FORMAL_BOUNDED"]; baseline_metric=next(r for r in pooled if r["arm"]=="B0_FORMAL_BOUNDED")
    comparisons=[]; supported=[]
    for arm in ARMS:
        metric=next(r for r in pooled if r["arm"]==arm)
        c_only,b_only,p=mcnemar_exact((arms[arm]>=.5)==y,(baseline>=.5)==y)
        noninferior=sum(next(r for r in per_fold if r["arm"]==arm and int(r["fold"])==f)["BA"] >= next(r for r in per_fold if r["arm"]=="B0_FORMAL_BOUNDED" and int(r["fold"])==f)["BA"] for f in range(1,6))
        row={"arm":arm,"BA_gain_pp_vs_bounded":100*(metric["BA"]-baseline_metric["BA"]),"candidate_only_correct":c_only,"bounded_only_correct":b_only,
             "paired_net":c_only-b_only,"mcnemar_exact_p":p,"noninferior_folds":noninferior,"broken_not_higher":metric["broken"]<=baseline_metric["broken"]}
        comparisons.append(row)
        if metric["net"]>baseline_metric["net"] and metric["broken"]<=baseline_metric["broken"] and noninferior>=3: supported.append(arm)
    write_csv(OUTPUT/"paired_vs_bounded.csv",comparisons)
    status="REMOVING_BOUNDED_CORRECTION_SUPPORTED_ON_SOURCE" if supported else "REMOVING_BOUNDED_CORRECTION_NOT_SUPPORTED"
    decision={"status":status,"supported_arms":supported,"future_rows_loaded":0,"formal_bounded":baseline_metric,
              "comparisons":comparisons,"future_authorized":bool(supported),"note":"Source-only ablation; future cannot select the arm."}
    dump_json(OUTPUT/"final_decision.json",decision)
    print("SVM_C43_REMOVE_BOUNDED_SOURCE_DONE",flush=True)
    for row in pooled: print(f"{row['arm']}: BA={row['BA']:.6f} AUC={row['AUC']:.6f} fixed={row['fixed']} broken={row['broken']} net={row['net']}",flush=True)


if __name__=="__main__":main()
