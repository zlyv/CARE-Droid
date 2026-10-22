#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
from ablation_common import *

DIRECT_METHOD="P31SVM__SVMRoute__A0_ROLE_C43_RQ__DirectStage2"

def side_auc(y,base,route,score):
    pb=route&(base<.5); pm=route&(base>=.5)
    return {"FN_vs_TN_AUC":auc(y[pb],score[pb]),"TP_vs_FP_AUC":auc(y[pm],score[pm]),
      "side_FN":int(np.sum(pb&(y==1))),"side_TN":int(np.sum(pb&(y==0))),
      "side_TP":int(np.sum(pm&(y==1))),"side_FP":int(np.sum(pm&(y==0)))}

def main():
    formal=formal_rows(); ids=[norm(r["sha256"]) for r in formal]
    y=np.asarray([int(float(r["label"])) for r in formal]); fold=np.asarray([int(float(r["fold"])) for r in formal])
    base=np.asarray([float(r["stage1_prob"]) for r in formal]); route=np.asarray([int(float(r["routed"])) for r in formal],bool)
    direct_rows=read_csv(DIRECT_REFERENCE); direct_by={norm(r["sha256"]):r for r in direct_rows}
    direct=np.asarray([float(direct_by[sid][DIRECT_METHOD+"__prob"]) for sid in ids])
    probabilities={}; expert={}; pooled=[]; per_fold=[]
    for arm in ARMS:
        method=METHOD[arm]; rows=read_csv(OUTPUT/"source"/arm/"source_predictions.csv"); by={norm(r["sha256"]):r for r in rows}
        if set(by)!=set(ids): raise RuntimeError("BLOCKED_BY_CONTRACT: SHA mismatch for "+arm)
        ordered=[by[sid] for sid in ids]; prob=np.asarray([float(r[method+"__prob"]) for r in ordered])
        exp=np.asarray([float(r[method+"__expert_prob"]) for r in ordered]); arm_route=np.asarray([int(float(r["A25__routed"])) for r in ordered],bool)
        arm_base=np.asarray([float(r["Global24__prob"]) for r in ordered])
        if not np.array_equal(arm_route,route) or not np.allclose(arm_base,base,atol=1e-12):
            raise RuntimeError("BLOCKED_BY_CONTRACT: Stage-I/router changed for "+arm)
        probabilities[arm]=prob; expert[arm]=exp
        pooled.append({"arm":arm,**metrics(y,base,prob,route),**side_auc(y,base,route,exp)})
        for f in range(1,6):
            s=fold==f; per_fold.append({"arm":arm,"fold":f,**metrics(y[s],base[s],prob[s],route[s]),**side_auc(y[s],base[s],route[s],exp[s])})
    write_csv(OUTPUT/"source_results.csv",pooled); write_csv(OUTPUT/"source_fold_results.csv",per_fold)
    full="R11_ROLE_QUERY_READOUT"; full_metric=next(r for r in pooled if r["arm"]==full)
    replay_diff=float(np.max(np.abs(probabilities[full]-direct)))
    replay_pass=replay_diff<=1e-4 and (full_metric["fixed"],full_metric["broken"],full_metric["net"])==(49,17,32)
    if not replay_pass: raise RuntimeError("BLOCKED_BY_CONTRACT: R11 did not reproduce Direct-C43")
    comparisons=[]
    for arm in ARMS:
        metric=next(r for r in pooled if r["arm"]==arm); c,b,p=mcnemar_exact((probabilities[arm]>=.5)==y,(probabilities[full]>=.5)==y)
        noninferior=sum(next(r for r in per_fold if r["arm"]==arm and r["fold"]==f)["BA"] >= next(r for r in per_fold if r["arm"]==full and r["fold"]==f)["BA"]-.005 for f in range(1,6))
        comparisons.append({"arm":arm,"delta_BA_vs_R11":metric["BA"]-full_metric["BA"],"delta_net_vs_R11":metric["net"]-full_metric["net"],
          "candidate_only_correct":c,"R11_only_correct":b,"mcnemar_exact_p":p,"folds_BA_within_0.005_of_R11":noninferior})
    write_csv(OUTPUT/"paired_vs_R11.csv",comparisons)
    by={r["arm"]:r for r in pooled}; fby={(r["arm"],r["fold"]):r for r in per_fold}
    def effect(metric,a,b,c,d): return .5*((by[a][metric]-by[b][metric])+(by[c][metric]-by[d][metric]))
    q_fn=effect("FN_vs_TN_AUC","R11_ROLE_QUERY_READOUT","R01_ROLE_READOUT_ONLY","R10_ROLE_QUERY_ONLY","R00_ROLE_NEITHER")
    q_tp=effect("TP_vs_FP_AUC","R11_ROLE_QUERY_READOUT","R01_ROLE_READOUT_ONLY","R10_ROLE_QUERY_ONLY","R00_ROLE_NEITHER")
    r_fn=effect("FN_vs_TN_AUC","R11_ROLE_QUERY_READOUT","R10_ROLE_QUERY_ONLY","R01_ROLE_READOUT_ONLY","R00_ROLE_NEITHER")
    r_tp=effect("TP_vs_FP_AUC","R11_ROLE_QUERY_READOUT","R10_ROLE_QUERY_ONLY","R01_ROLE_READOUT_ONLY","R00_ROLE_NEITHER")
    fold_effects=[]
    for f in range(1,6):
        def fe(metric,a,b,c,d): return .5*((fby[(a,f)][metric]-fby[(b,f)][metric])+(fby[(c,f)][metric]-fby[(d,f)][metric]))
        fold_effects.append({"fold":f,"query_FN_effect":fe("FN_vs_TN_AUC","R11_ROLE_QUERY_READOUT","R01_ROLE_READOUT_ONLY","R10_ROLE_QUERY_ONLY","R00_ROLE_NEITHER"),
          "query_TP_effect":fe("TP_vs_FP_AUC","R11_ROLE_QUERY_READOUT","R01_ROLE_READOUT_ONLY","R10_ROLE_QUERY_ONLY","R00_ROLE_NEITHER"),
          "readout_FN_effect":fe("FN_vs_TN_AUC","R11_ROLE_QUERY_READOUT","R10_ROLE_QUERY_ONLY","R01_ROLE_READOUT_ONLY","R00_ROLE_NEITHER"),
          "readout_TP_effect":fe("TP_vs_FP_AUC","R11_ROLE_QUERY_READOUT","R10_ROLE_QUERY_ONLY","R01_ROLE_READOUT_ONLY","R00_ROLE_NEITHER")})
    write_csv(OUTPUT/"factorial_fold_effects.csv",fold_effects)
    q_folds=sum(r["query_FN_effect"]>0 and r["query_TP_effect"]>0 for r in fold_effects)
    r_folds=sum(r["readout_FN_effect"]>0 and r["readout_TP_effect"]>0 for r in fold_effects)
    query_supported=bool(q_fn>=.03 and q_tp>=.03 and q_folds>=3)
    readout_supported=bool(r_fn>=.03 and r_tp>=.03 and r_folds>=3)
    factorial={"query":{"FN_effect":q_fn,"TP_effect":q_tp,"both_positive_folds":q_folds,"supported":query_supported},
      "readout":{"FN_effect":r_fn,"TP_effect":r_tp,"both_positive_folds":r_folds,"supported":readout_supported},
      "interaction":{"FN":by["R11_ROLE_QUERY_READOUT"]["FN_vs_TN_AUC"]-by["R10_ROLE_QUERY_ONLY"]["FN_vs_TN_AUC"]-by["R01_ROLE_READOUT_ONLY"]["FN_vs_TN_AUC"]+by["R00_ROLE_NEITHER"]["FN_vs_TN_AUC"],
        "TP":by["R11_ROLE_QUERY_READOUT"]["TP_vs_FP_AUC"]-by["R10_ROLE_QUERY_ONLY"]["TP_vs_FP_AUC"]-by["R01_ROLE_READOUT_ONLY"]["TP_vs_FP_AUC"]+by["R00_ROLE_NEITHER"]["TP_vs_FP_AUC"]}}
    candidates=[]
    for arm in ARMS[1:]:
        m=by[arm]; cmp=next(r for r in comparisons if r["arm"]==arm)
        if m["fixed"]>=44 and m["broken"]<=12 and m["net"]>=32 and cmp["folds_BA_within_0.005_of_R11"]>=4: candidates.append(arm)
    if candidates: status="ROLEPATH_SIMPLIFICATION_CANDIDATE_FOUND"
    elif query_supported and readout_supported: status="BOTH_ROLEGRAPH_PATHS_CONTRIBUTE"
    elif query_supported: status="ROLEGRAPH_QUERY_PATH_CONTRIBUTES"
    elif readout_supported: status="ROLEGRAPH_READOUT_PATH_CONTRIBUTES"
    else: status="ROLEGRAPH_PATH_FACTORIAL_INCONCLUSIVE"
    next_action="freeze and independently validate the source-qualified simplification" if candidates else "stop path pruning; remaining broken is not explained by removing either RoleGraph path"
    decision={"status":status,"R11_replay_max_abs_difference":replay_diff,"R11_replay_pass":replay_pass,
      "factorial_effects":factorial,"simplification_candidates":candidates,"source_results":{r["arm"]:r for r in pooled},
      "next_action":next_action,"future_rows_loaded":0}
    dump_json(OUTPUT/"factorial_effects.json",factorial); dump_json(OUTPUT/"final_decision.json",decision)
    print("DIRECT_C43_ROLEPATH_FACTORIAL_STATUS="+status,flush=True)
    for r in pooled: print(f"{r['arm']}: BA={r['BA']:.6f} FN_AUC={r['FN_vs_TN_AUC']:.6f} TP_AUC={r['TP_vs_FP_AUC']:.6f} fixed={r['fixed']} broken={r['broken']} net={r['net']}",flush=True)

if __name__=="__main__":main()
