#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from common import (ARMS, DIRECT_PREDICTIONS, FOLDS, OUTPUT, SEED, binary_auc, bootstrap_auc, dump_json,
                    finite, host_summary, load_source, match_pairs, norm, read_csv, write_csv)


def fit_transform_c43(train, test, raw):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler(); train_z = scaler.fit_transform(raw[train]); test_z = scaler.transform(raw[test])
    width = min(24, len(train) - 1, train_z.shape[1]); pca = PCA(n_components=width, svd_solver="full")
    return pca.fit_transform(train_z), pca.transform(test_z), float(pca.explained_variance_ratio_.sum())


def fit_pair_ranker(train_indices, train_x, pairs, seed):
    from sklearn.linear_model import LogisticRegression
    position = {int(value): i for i, value in enumerate(train_indices)}; diffs = []
    for malware, benign, _, _ in pairs:
        d = train_x[position[malware]] - train_x[position[benign]]; diffs.extend([d, -d])
    x = np.asarray(diffs, float); target = np.tile([1, 0], len(pairs))
    model = LogisticRegression(C=.25, solver="liblinear", fit_intercept=False, max_iter=5000, random_state=int(seed))
    model.fit(x, target); return model


def fit_direct(train_x, train_y, seed):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=.25, class_weight="balanced", solver="liblinear", max_iter=5000, random_state=int(seed))
    model.fit(train_x, train_y); return model


def standardized_match_features(train, target, p31, base, route_score):
    p = np.sqrt(np.maximum(p31, 0.0)); clipped = np.clip(base, 1e-6, 1-1e-6); logit = np.log(clipped/(1-clipped))
    raw = np.column_stack([p, logit, route_score]); mean=raw[train].mean(0); std=raw[train].std(0); std[std<1e-8]=1
    return (raw-mean)/std


def eval_direction(y, pred, route, score, selected):
    chosen = np.asarray(selected, bool) & route
    benign_side = chosen & (~pred); malware_side = chosen & pred
    return {"FN_vs_TN_AUC": binary_auc(y[benign_side], score[benign_side]),
            "TP_vs_FP_AUC": binary_auc(y[malware_side], score[malware_side]),
            "benign_side_n": int(benign_side.sum()), "benign_side_FN": int(y[benign_side].sum()),
            "malware_side_n": int(malware_side.sum()), "malware_side_TP": int(y[malware_side].sum())}


def main():
    rows, ids, y, fold, member, base, route_score, route, p31, bags = load_source(); pred=base>=.5
    c43_raw, host_count = host_summary(bags); count_raw=np.log1p(host_count).reshape(-1,1)
    scores={arm:np.full(len(y),np.nan) for arm in ARMS}; pair_rows=[]; fold_rows=[]; explained=[]
    permutation_scores=np.full((100,len(y)),np.nan,float)
    for current in FOLDS:
        train=np.flatnonzero(fold!=current); test=np.flatnonzero(fold==current); match_x=standardized_match_features(train,test,p31,base,route_score)
        real_pairs=match_pairs(train,y,pred,match_x,False,SEED+current); random_pairs=match_pairs(train,y,pred,match_x,True,SEED+100+current)
        train_c43,test_c43,variance=fit_transform_c43(train,test,c43_raw); explained.append({"fold":current,"PCA24_variance":variance})
        p31_mean=p31[train].mean(0); p31_std=p31[train].std(0); p31_std[p31_std<1e-8]=1
        train_p31=(p31[train]-p31_mean)/p31_std; test_p31=(p31[test]-p31_mean)/p31_std
        count_mean=count_raw[train].mean(0); count_std=count_raw[train].std(0); count_std[count_std<1e-8]=1
        train_count=(count_raw[train]-count_mean)/count_std; test_count=(count_raw[test]-count_mean)/count_std
        models={"C43_MATCHED_REAL":fit_pair_ranker(train,train_c43,real_pairs,SEED+current),
                "C43_RANDOM_PAIR":fit_pair_ranker(train,train_c43,random_pairs,SEED+100+current),
                "C43_UNMATCHED":fit_direct(train_c43,y[train],SEED+200+current),
                "P31_MATCHED":fit_pair_ranker(train,train_p31,real_pairs,SEED+300+current),
                "COUNT_ONLY_MATCHED":fit_pair_ranker(train,train_count,real_pairs,SEED+400+current)}
        test_values={"C43_MATCHED_REAL":test_c43,"C43_RANDOM_PAIR":test_c43,"C43_UNMATCHED":test_c43,
                     "P31_MATCHED":test_p31,"COUNT_ONLY_MATCHED":test_count}
        for arm in ARMS: scores[arm][test]=models[arm].decision_function(test_values[arm])
        validation_pairs=match_pairs(test,y,pred,match_x,False,SEED+500+current)
        for malware,benign,side,distance in validation_pairs:
            pair_rows.append({"fold":current,"pair_scope":"outer_validation_audit","stage1_pred":side,"malware_sha256":ids[malware],"benign_sha256":ids[benign],
                              "match_distance":distance, **{arm+"_margin":float(scores[arm][malware]-scores[arm][benign]) for arm in ARMS}})
        selected=fold==current
        for arm in ARMS:
            value=eval_direction(y,pred,route,scores[arm],selected)
            local_pairs=[row for row in pair_rows if row["fold"]==current]
            value.update({"arm":arm,"fold":current,"matched_pair_accuracy":float(np.mean([row[arm+"_margin"]>0 for row in local_pairs])) if local_pairs else float("nan")})
            fold_rows.append(value)
        rng=np.random.RandomState(SEED+900+current)
        for repeat in range(100):
            shuffled=y.copy(); local_labels=shuffled[train].copy(); rng.shuffle(local_labels); shuffled[train]=local_labels
            null_pairs=match_pairs(train,shuffled,pred,match_x,False,SEED+repeat+current*1000)
            if not null_pairs: continue
            null_model=fit_pair_ranker(train,train_c43,null_pairs,SEED+repeat+current*2000)
            permutation_scores[repeat,test]=null_model.decision_function(test_c43)
        for malware,benign,side,distance in real_pairs:
            pair_rows.append({"fold":current,"pair_scope":"outer_train","stage1_pred":side,"malware_sha256":ids[malware],"benign_sha256":ids[benign],"match_distance":distance})
    for arm in ARMS:
        if not np.isfinite(scores[arm]).all(): raise RuntimeError("BLOCKED_BY_CONTRACT: incomplete OOF score "+arm)
    direct=read_csv(DIRECT_PREDICTIONS); method="P31SVM__SVMRoute__A0_ROLE_C43_RQ__DirectStage2"
    direct_map={norm(row["sha256"]):float(row[method+"__prob"]) for row in direct}
    if set(direct_map) != set(ids): raise RuntimeError("BLOCKED_BY_CONTRACT: existing direct-C43 SHA set changed")
    direct_score=np.asarray([np.log(np.clip(direct_map[sid],1e-8,1-1e-8)/np.clip(1-direct_map[sid],1e-8,1-1e-8)) for sid in ids])
    all_scores={**scores,"EXISTING_DIRECT_C43":direct_score}; summaries={}
    for ordinal,(arm,score) in enumerate(all_scores.items()):
        chosen=route; benign=chosen&(~pred); malware=chosen&pred
        if arm == "EXISTING_DIRECT_C43":
            direct_folds=[eval_direction(y,pred,route,score,fold==current) for current in FOLDS]
            fn_folds=sum(finite(row["FN_vs_TN_AUC"]) and row["FN_vs_TN_AUC"]>.5 for row in direct_folds)
            tp_folds=sum(finite(row["TP_vs_FP_AUC"]) and row["TP_vs_FP_AUC"]>.5 for row in direct_folds)
        else:
            fn_folds=sum(finite(row["FN_vs_TN_AUC"]) and row["FN_vs_TN_AUC"]>.5 for row in fold_rows if row["arm"]==arm)
            tp_folds=sum(finite(row["TP_vs_FP_AUC"]) and row["TP_vs_FP_AUC"]>.5 for row in fold_rows if row["arm"]==arm)
        summaries[arm]={"FN_vs_TN":bootstrap_auc(y[benign],score[benign],SEED+ordinal),
                        "TP_vs_FP":bootstrap_auc(y[malware],score[malware],SEED+100+ordinal),
                        "folds_FN_above_random":fn_folds,"folds_TP_above_random":tp_folds}
    permutation_values={"FN_vs_TN_AUC":[],"TP_vs_FP_AUC":[]}
    for repeat in range(100):
        result=eval_direction(y,pred,route,permutation_scores[repeat],np.ones(len(y),bool))
        for key in permutation_values:
            if finite(result[key]): permutation_values[key].append(result[key])
    null95={key:(float(np.quantile(value,.95)) if value else float("nan")) for key,value in permutation_values.items()}
    real=summaries["C43_MATCHED_REAL"]
    null_arms=("C43_RANDOM_PAIR","C43_UNMATCHED","P31_MATCHED","COUNT_ONLY_MATCHED","EXISTING_DIRECT_C43")
    best_fn=max(summaries[a]["FN_vs_TN"]["auc"] for a in null_arms); best_tp=max(summaries[a]["TP_vs_FP"]["auc"] for a in null_arms)
    criteria={"FN_AUC>=0.60":real["FN_vs_TN"]["auc"]>=.60,"TP_AUC>=0.60":real["TP_vs_FP"]["auc"]>=.60,
              "FN>=best_null+0.03":real["FN_vs_TN"]["auc"]>=best_fn+.03,"TP>=best_null+0.03":real["TP_vs_FP"]["auc"]>=best_tp+.03,
              "FN>label_shuffle95+0.03":finite(null95["FN_vs_TN_AUC"]) and real["FN_vs_TN"]["auc"]>=null95["FN_vs_TN_AUC"]+.03,
              "TP>label_shuffle95+0.03":finite(null95["TP_vs_FP_AUC"]) and real["TP_vs_FP"]["auc"]>=null95["TP_vs_FP_AUC"]+.03,
              "FN_direction_in>=3folds":real["folds_FN_above_random"]>=3,"TP_direction_in>=3folds":real["folds_TP_above_random"]>=3}
    if all(criteria.values()): status="CONDITIONAL_MATCHED_C43_SUPPORTED"
    elif ((criteria["FN_AUC>=0.60"] and criteria["FN>=best_null+0.03"]) or (criteria["TP_AUC>=0.60"] and criteria["TP>=best_null+0.03"])): status="CONDITIONAL_MATCHED_C43_PARTIALLY_SUPPORTED"
    else: status="CONDITIONAL_MATCHED_C43_NOT_SUPPORTED"
    decision={"status":status,"criteria":criteria,"results":summaries,"best_null":{"FN":best_fn,"TP":best_tp},
              "label_shuffle_95pct":null95,"future_rows_loaded":0,"end_to_end_run":False,
              "routed_sample_counts":{"total":int(route.sum()),"predicted_benign":int(np.sum(route&(~pred))),
                  "predicted_malware":int(np.sum(route&pred)),"FN":int(np.sum(route&(~pred)&(y==1))),
                  "TN":int(np.sum(route&(~pred)&(y==0))),"TP":int(np.sum(route&pred&(y==1))),"FP":int(np.sum(route&pred&(y==0)))},
              "pair_counts":{"outer_train_total":sum(row.get("pair_scope")=="outer_train" for row in pair_rows),
                  "outer_validation_audit_total":sum(row.get("pair_scope")=="outer_validation_audit" for row in pair_rows)},
              "next_action":"freeze source result; only a fully supported result may authorize a separate integration experiment"}
    dump_json(OUTPUT/"probe_results.json",decision); dump_json(OUTPUT/"final_decision.json",decision)
    write_csv(OUTPUT/"source_oof_scores.csv",[{"sha256":ids[i],"fold":int(fold[i]),"label":int(y[i]),"stage1_prob":float(base[i]),"stage1_pred":int(pred[i]),"routed":int(route[i]),**{a:float(s[i]) for a,s in all_scores.items()}} for i in range(len(y))])
    write_csv(OUTPUT/"fold_results.csv",fold_rows); write_csv(OUTPUT/"matched_pairs.csv",pair_rows); write_csv(OUTPUT/"pca_audit.csv",explained)
    print("C43_CONDITIONAL_MATCHED_SOURCE_STATUS="+status,flush=True)
    print("REAL FN_AUC={:.6f} TP_AUC={:.6f} best_null_FN={:.6f} best_null_TP={:.6f}".format(real["FN_vs_TN"]["auc"],real["TP_vs_FP"]["auc"],best_fn,best_tp),flush=True)


if __name__ == "__main__": main()
