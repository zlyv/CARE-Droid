#!/usr/bin/env python3
from __future__ import annotations
import argparse, math
import numpy as np
from common import *
from tb_frs import build_fold,reduce,WIDTH

def fit_score(train_x,train_y,test_x,seed):
    from sklearn.linear_model import LogisticRegression
    model=LogisticRegression(C=.25,class_weight="balanced",solver="liblinear",max_iter=5000,random_state=int(seed))
    model.fit(train_x,train_y); return model.decision_function(test_x)

def direction(y,base,route,score,selected=None):
    pred=np.asarray(base)>=.5; chosen=np.asarray(route,bool) if selected is None else np.asarray(route,bool)&np.asarray(selected,bool)
    left=chosen&(~pred); right=chosen&pred
    return {"FN_vs_TN_AUC":auc(y[left],score[left]),"TP_vs_FP_AUC":auc(y[right],score[right]),
      "predicted_benign_n":int(left.sum()),"FN":int(np.sum(y[left]==1)),"predicted_malware_n":int(right.sum()),"TP":int(np.sum(y[right]==1))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--null-repeats",type=int,default=200); cli=ap.parse_args()
    rows,ids,y,fold,base,route,bags,direct_prob,direct_field=load_source(); host_count=np.asarray(list(map(len,bags)),float)
    scores={a:np.full(len(y),np.nan) for a in ARMS}; null={a:np.full((cli.null_repeats,len(y)),np.nan) for a in ("B2_SIGNED_FIRST","B3_SIGNED_FIRST_SECOND")}
    fold_audit=[]; pca_audit=[]
    for current in FOLDS:
        train=np.flatnonzero(fold!=current); test=np.flatnonzero(fold==current)
        raw_train,raw_test,audit=build_fold([bags[i] for i in train],[bags[i] for i in test],y[train],ids[train],ids[test],SEED+current*100)
        fold_audit.append({"fold":current,**audit})
        for ordinal,arm in enumerate(ARMS):
            tx,vx,variance=reduce(raw_train[arm],raw_test[arm],SEED+current*1000+ordinal,WIDTH)
            scores[arm][test]=fit_score(tx,y[train],vx,SEED+current*2000+ordinal)
            pca_audit.append({"fold":current,"arm":arm,"raw_width":int(raw_train[arm].shape[1]),"model_width":WIDTH,"variance_retained":variance})
            if arm in null:
                rng=np.random.RandomState(SEED+current*10000+ordinal)
                for repeat in range(cli.null_repeats):
                    shuffled=y[train].copy(); rng.shuffle(shuffled)
                    null[arm][repeat,test]=fit_score(tx,shuffled,vx,SEED+repeat+current*10000+ordinal)
        print("TBFRS_OUTER_FOLD_DONE fold=%d/5"%current,flush=True)
    for arm in ARMS:
        if not np.isfinite(scores[arm]).all():raise RuntimeError("BLOCKED_BY_CONTRACT: incomplete OOF "+arm)
    direct_score=np.log(np.clip(direct_prob,1e-8,1-1e-8)/np.clip(1-direct_prob,1e-8,1-1e-8))
    all_scores={"B0_DIRECT_C43_REFERENCE":direct_score,**scores}; result={}; fold_rows=[]
    from scipy.stats import spearmanr
    for ordinal,(arm,score) in enumerate(all_scores.items()):
        d=direction(y,base,route,score); pred=base>=.5; left=route&(~pred); right=route&pred
        folds=[]
        for current in FOLDS:
            one=direction(y,base,route,score,fold==current); one.update({"arm":arm,"fold":current}); fold_rows.append(one); folds.append(one)
        corr=float(spearmanr(score[route],np.log1p(host_count[route])).statistic)
        result[arm]={**d,"FN_vs_TN":bootstrap(y[left],score[left],SEED+ordinal),"TP_vs_FP":bootstrap(y[right],score[right],SEED+100+ordinal),
          "folds_FN_above_random":sum(finite(x["FN_vs_TN_AUC"]) and x["FN_vs_TN_AUC"]>.5 for x in folds),
          "folds_TP_above_random":sum(finite(x["TP_vs_FP_AUC"]) and x["TP_vs_FP_AUC"]>.5 for x in folds),
          "host_count_spearman":None if not finite(corr) else corr}
    null95={}
    for arm,values in null.items():
        left=[]; right=[]
        for repeat in range(cli.null_repeats):
            d=direction(y,base,route,values[repeat]);
            if finite(d["FN_vs_TN_AUC"]):left.append(d["FN_vs_TN_AUC"])
            if finite(d["TP_vs_FP_AUC"]):right.append(d["TP_vs_FP_AUC"])
        null95[arm]={"FN_vs_TN_AUC":float(np.quantile(left,.95)),"TP_vs_FP_AUC":float(np.quantile(right,.95)),"repeats":min(len(left),len(right))}

    def support(arm,unsigned,baseline):
        r=result[arm]; u=result[unsigned]; b=result[baseline]; n=null95[arm]
        criteria={"FN_AUC>=0.60":r["FN_vs_TN_AUC"]>=.60,"TP_AUC>=0.60":r["TP_vs_FP_AUC"]>=.60,
          "FN>label_shuffle95+0.03":r["FN_vs_TN_AUC"]>=n["FN_vs_TN_AUC"]+.03,
          "TP>label_shuffle95+0.03":r["TP_vs_FP_AUC"]>=n["TP_vs_FP_AUC"]+.03,
          "FN>unsigned+0.03":r["FN_vs_TN_AUC"]>=u["FN_vs_TN_AUC"]+.03,
          "TP>unsigned+0.03":r["TP_vs_FP_AUC"]>=u["TP_vs_FP_AUC"]+.03,
          "FN>compressed":r["FN_vs_TN_AUC"]>b["FN_vs_TN_AUC"],"TP>compressed":r["TP_vs_FP_AUC"]>b["TP_vs_FP_AUC"],
          "FN_direction_in>=3folds":r["folds_FN_above_random"]>=3,"TP_direction_in>=3folds":r["folds_TP_above_random"]>=3,
          "abs_size_correlation<0.70":r["host_count_spearman"] is not None and abs(r["host_count_spearman"])<.70}
        return criteria,all(criteria.values())
    h2,h2ok=support("B2_SIGNED_FIRST","B2U_UNSIGNED_FIRST","B1_SOFT4")
    h3,h3ok=support("B3_SIGNED_FIRST_SECOND","B4_UNSIGNED_FIRST_SECOND","B2_SIGNED_FIRST")
    h1={"FN_soft>hard":result["B1_SOFT4"]["FN_vs_TN_AUC"]>result["H0_HARD4"]["FN_vs_TN_AUC"],
        "TP_soft>hard":result["B1_SOFT4"]["TP_vs_FP_AUC"]>result["H0_HARD4"]["TP_vs_FP_AUC"]}
    if h3ok:status="TB_FRS_SECOND_ORDER_INFORMATION_SUFFICIENT"; candidate="B3_SIGNED_FIRST_SECOND"
    elif h2ok:status="TB_FRS_FIRST_ORDER_INFORMATION_SUFFICIENT"; candidate="B2_SIGNED_FIRST"
    elif any(h1.values()):status="TB_FRS_PARTIAL_SOFT_ASSIGNMENT_SIGNAL"; candidate=None
    else:status="TB_FRS_INFORMATION_NOT_SUPPORTED"; candidate=None
    decision={"status":status,"candidate_for_separate_integration":candidate,"continue_to_end_to_end":candidate is not None,
      "hypotheses":{"H1_soft_assignment":h1,"H2_signed_first_order":h2,"H3_second_order":h3},"results":result,
      "label_shuffle_95pct":null95,"future_rows_loaded":0,"end_to_end_run":False,
      "decision_boundary":"This source-only probe cannot change Direct-C43, routing, threshold, or future protocol."}
    dump_json(OUTPUT/"probe_results.json",decision); dump_json(OUTPUT/"final_decision.json",decision)
    write_csv(OUTPUT/"fold_results.csv",fold_rows); write_csv(OUTPUT/"fold_prototype_audit.csv",fold_audit); write_csv(OUTPUT/"projection_audit.csv",pca_audit)
    write_csv(OUTPUT/"source_oof_scores.csv",[{"sha256":ids[i],"fold":int(fold[i]),"label":int(y[i]),"stage1_prob":float(base[i]),"routed":int(route[i]),"host_count":int(host_count[i]),**{a:float(s[i]) for a,s in all_scores.items()}} for i in range(len(y))])
    print("TBFRS_SOURCE_STATUS="+status,flush=True)
    for arm in ("H0_HARD4","B1_SOFT4","B2_SIGNED_FIRST","B3_SIGNED_FIRST_SECOND","B4_UNSIGNED_FIRST_SECOND","B5_COMPONENT_SHUFFLE","B6_CAPACITY_CONTROL"):
        r=result[arm]; print("%s FN_AUC=%.6f TP_AUC=%.6f corr=%s"%(arm,r["FN_vs_TN_AUC"],r["TP_vs_FP_AUC"],r["host_count_spearman"]),flush=True)
if __name__=="__main__":main()
