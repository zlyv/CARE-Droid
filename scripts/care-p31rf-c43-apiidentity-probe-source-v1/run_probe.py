#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from api_identity_core import (API_DIM, ROLE_N, FoldAPIProjector, build_api_bags,
    jaccard, norm_sha, read_csv, read_endpoint_cache, write_csv)

ROOT=Path("artifacts/care-p31rf-c43-apiidentity-probe-source-v1")
REF=Path("artifacts/care-p31-stage1-rf-selection-source-v1/preflight/C_RF_RFROUTE/source_predictions.csv")
PRED=Path("artifacts/care-p31-stage1-rf-selection-source-v1/arms/C_RF_RFROUTE/source_predictions.csv")
ENDPOINT=Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz")
CACHE=Path("artifacts/care-di25-compact43-relationquery-source-v1/cache/compact43_relation_source.npz")
ROUTER_INPUT=Path("artifacts/care-source-di25-reconstructed-v1/source_router_inputs.csv")
FINAL_PREFERRED="P31RF__RFRoute__P0Zero6C43RQ__Bounded__prob"
METHODS=("H0_ROLE","H1_ROLE_API","H1_ROLE_API_RAW","COUNT_ONLY")


def probability_to_logit(p):
    p=np.clip(np.asarray(p,float),1e-7,1-1e-7);return np.log(p/(1-p))


def find_final_column(rows):
    fields=list(rows[0])
    if FINAL_PREFERRED in fields:return FINAL_PREFERRED
    candidates=[x for x in fields if x.endswith("__prob") and "P31RF" in x and "Bounded" in x]
    if len(candidates)!=1:raise RuntimeError("BLOCKED_BY_CONTRACT: cannot identify frozen RF C43 probability: "+repr(candidates))
    return candidates[0]


def load_c43(ids):
    with np.load(CACHE,allow_pickle=False) as z:
        saved=np.asarray([norm_sha(v) for v in z["sha256"]])
        if set(saved.tolist())!=set(ids):raise RuntimeError("BLOCKED_BY_CONTRACT: C43 SHA cohort mismatch")
        offsets_key=next((k for k in z.files if "offset" in k.casefold() and z[k].shape==(800,)),None)
        if offsets_key is None:raise RuntimeError("BLOCKED_BY_CONTRACT: C43 offsets absent")
        offsets=np.asarray(z[offsets_key],int)
        values_key=next((k for k in z.files if z[k].ndim==2 and z[k].shape==(int(offsets[-1]),105)),None)
        if values_key is None:raise RuntimeError("BLOCKED_BY_CONTRACT: N x 105 C43 matrix absent")
        values=np.asarray(z[values_key],float)
        real=np.asarray(z["real_host_count"],int) if "real_host_count" in z else np.diff(offsets)
    by_sha={saved[i]:values[offsets[i]:offsets[i]+real[i],:43] for i in range(799)}
    return [by_sha[sha] for sha in ids], offsets_key, values_key


def role_apk_matrix(bags):
    out=np.zeros((len(bags),ROLE_N),float)
    for i,bag in enumerate(bags):
        if bag:out[i]=np.mean([host["role"] for host in bag],axis=0)
    return out


def fit_probe(x,y,train,val,seed):
    model=Pipeline([("scale",StandardScaler()),("model",LogisticRegression(
        C=1.0,solver="liblinear",class_weight="balanced",max_iter=5000,random_state=int(seed)))])
    model.fit(x[train],y[train]);return np.clip(model.predict_proba(x[val])[:,1],1e-7,1-1e-7)


def safe_auc(y,p):
    return float(roc_auc_score(y,p)) if len(np.unique(y))==2 else float("nan")


def safe_ba(y,p):
    return float(balanced_accuracy_score(y,p>=.5)) if len(y) else float("nan")


def safe_f1(y,p):
    return float(f1_score(y,p>=.5,zero_division=0)) if len(y) else float("nan")


def metric(y,p):
    return {"n":int(len(y)),"BA":safe_ba(y,p),"AUC":safe_auc(y,p),"F1":safe_f1(y,p)}


def direction_row(name,y,base,route,score,fold):
    error=route&((base>=.5)!=y);truth=np.where(y==1,1.,-1.);signed=probability_to_logit(score)
    aligned=signed*truth;correct=aligned>1e-12;amb=np.abs(aligned)<=1e-12
    fn=error&(y==1);fp=error&(y==0)
    fold_values=[]
    for value in range(1,6):
        s=error&(fold==value);fold_values.append(float(np.mean(correct[s])) if np.any(s) else float("nan"))
    clean=[v for v in fold_values if np.isfinite(v)]
    return {"Representation":name,"RoutedErrors":int(error.sum()),"FN":int(fn.sum()),"FP":int(fp.sum()),
      "OverallDirectionAcc":float(np.mean(correct[error])),"FNDirectionAcc":float(np.mean(correct[fn])),
      "FPDirectionAcc":float(np.mean(correct[fp])),"WrongDirection":int(np.sum(error&~correct&~amb)),
      "Ambiguous":int(np.sum(error&amb)),"CorrectDirection":int(np.sum(error&correct)),
      "MeanAligned":float(np.mean(aligned[error])),"MedianAligned":float(np.median(aligned[error])),
      "FNMeanAligned":float(np.mean(aligned[fn])),"FPMeanAligned":float(np.mean(aligned[fp])),
      "FoldMean":float(np.mean(clean)),"FoldStd":float(np.std(clean)),"DirectionAUC":safe_auc(y[error],score[error])}


def exact_paired(y,base,route,left,right,name):
    s=route&((base>=.5)!=y);a=(left[s]>=.5)==y[s];b=(right[s]>=.5)==y[s]
    n01=int(np.sum(a&~b));n10=int(np.sum(~a&b));n=n01+n10
    return {"comparison":name,"candidate_only_correct":n01,"control_only_correct":n10,
      "paired_net":n01-n10,"exact_p":float(binomtest(min(n01,n10),n,.5).pvalue) if n else 1.0}


def extra_size_counts(ids):
    result={sha:{} for sha in ids}
    if not ROUTER_INPUT.exists():return result,[]
    rows=read_csv(ROUTER_INPUT);fields=list(rows[0]) if rows else []
    def choose(tokens):
        candidates=[]
        for field in fields:
            low=field.casefold()
            if all(token in low for token in tokens):candidates.append(field)
        return candidates[0] if candidates else None
    selected={"apk_method_count":choose(("method","count")),"audit_graph_node_count":choose(("graph","node")),
              "audit_graph_edge_count":choose(("graph","edge"))}
    for row in rows:
        sha=norm_sha(row.get("sha256"))
        if sha not in result:continue
        for name,field in selected.items():
            if field:
                try:result[sha][name]=float(row[field])
                except (ValueError,TypeError):pass
    return result,[{"audit_name":k,"source_column":v or "UNAVAILABLE"} for k,v in selected.items()]


def write_host_audit(path,records):
    path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,"wt",encoding="utf-8") as h:
        for row in records:h.write(json.dumps(row,separators=(",",":"))+"\n")


def collapse_audit(hosts):
    errors=[row for row in hosts if row["stage1_error"]==1]
    if len(errors)<2:return [],{"host_rows":len(errors)}
    matrix=np.asarray([row["c43"] for row in errors],float)
    model=NearestNeighbors(n_neighbors=min(25,len(errors)),metric="cosine",algorithm="brute").fit(matrix)
    dist,index=model.kneighbors(matrix)
    details=[]
    for i,row in enumerate(errors):
        chosen=None
        for d,j in zip(dist[i,1:],index[i,1:]):
            if errors[int(j)]["sha256"]!=row["sha256"]:
                chosen=(float(d),errors[int(j)]);break
        if chosen is None:continue
        d,other=chosen
        details.append({"sha256":row["sha256"],"host_index":row["host_index"],
          "neighbor_sha256":other["sha256"],"neighbor_host_index":other["host_index"],
          "c43_cosine_similarity":1-d,"correction_sign_disagrees":int(row["truth_sign"]!=other["truth_sign"]),
          "label_disagrees":int(row["label"]!=other["label"]),"same_role":int(row["role_ids"]==other["role_ids"]),
          "api_jaccard":jaccard(row["api_ids"],other["api_ids"]),"api_sets_differ":int(set(row["api_ids"])!=set(other["api_ids"]))})
    groups=defaultdict(list)
    for row in errors:groups[tuple(row["role_ids"])].append(row)
    eligible=0;different=0
    for row in errors:
        opposite=[x for x in groups[tuple(row["role_ids"])] if x["sha256"]!=row["sha256"] and x["truth_sign"]!=row["truth_sign"]]
        if opposite:
            eligible+=1
            if any(set(x["api_ids"])!=set(row["api_ids"]) for x in opposite):different+=1
    summary={"error_host_rows":len(errors),"nearest_neighbor_rows":len(details),
      "nearest_neighbor_label_disagreement":float(np.mean([r["label_disagrees"] for r in details])) if details else float("nan"),
      "nearest_neighbor_correction_sign_disagreement":float(np.mean([r["correction_sign_disagrees"] for r in details])) if details else float("nan"),
      "nearest_neighbor_same_role_fraction":float(np.mean([r["same_role"] for r in details])) if details else float("nan"),
      "nearest_neighbor_api_sets_differ_fraction":float(np.mean([r["api_sets_differ"] for r in details])) if details else float("nan"),
      "nearest_neighbor_mean_api_jaccard":float(np.mean([r["api_jaccard"] for r in details])) if details else float("nan"),
      "exact_role_with_opposite_direction_hosts":eligible,"exact_role_opposite_direction_different_api_hosts":different,
      "exact_role_opposite_direction_different_api_fraction":float(different/eligible) if eligible else float("nan")}
    return details,summary


def main():
    ref=read_csv(REF);pred_raw=read_csv(PRED);ids=[norm_sha(r["sha256"]) for r in ref]
    pred_by_sha={norm_sha(r["sha256"]):r for r in pred_raw}
    if set(ids)!=set(pred_by_sha):raise RuntimeError("BLOCKED_BY_CONTRACT: prediction cohort mismatch")
    pred=[pred_by_sha[sha] for sha in ids]
    y=np.asarray([int(float(r["label"])) for r in ref]);fold=np.asarray([int(float(r["fold"])) for r in ref])
    base=np.asarray([float(r["Global24__prob"]) for r in ref]);route=np.asarray([int(float(r["A25__routed"]))==1 for r in ref])
    final_col=find_final_column(pred);final=np.asarray([float(r[final_col]) for r in pred])
    current_delta=probability_to_logit(final)-probability_to_logit(base);current_score=1/(1+np.exp(-current_delta))
    endpoint,load_seconds=read_endpoint_cache(ENDPOINT)
    bags,audits,failures,reconstruction_seconds=build_api_bags(ref,endpoint)
    c43,offsets_key,values_key=load_c43(ids)
    for i,(bag,matrix) in enumerate(zip(bags,c43)):
        if len(bag)!=len(matrix):raise RuntimeError(f"BLOCKED_BY_CONTRACT: Host order/count mismatch {ids[i]} reconstructed={len(bag)} c43={len(matrix)}")
    role=role_apk_matrix(bags);oof={name:np.full(799,np.nan) for name in METHODS};feature_norm={name:np.full(799,np.nan) for name in METHODS}
    unknown_rate=np.full(799,np.nan);vocab_audit=[];host_records=[]
    for f in range(1,6):
        train=np.flatnonzero(fold!=f);val=np.flatnonzero(fold==f)
        projector=FoldAPIProjector(seed=17000+f).fit([bags[i] for i in train])
        tr=projector.transform([bags[i] for i in train]);va=projector.transform([bags[i] for i in val])
        tr_raw=projector.transform([bags[i] for i in train],raw_count=True);va_raw=projector.transform([bags[i] for i in val],raw_count=True)
        xtr={"H0_ROLE":np.c_[tr["role"],np.zeros((len(train),API_DIM))],
          "H1_ROLE_API":np.c_[tr["role"],tr["api"]],"H1_ROLE_API_RAW":np.c_[tr["role"],tr_raw["api"]],
          "COUNT_ONLY":np.c_[tr["host_count"],tr["api_total"],np.asarray([audits[i]["unique_api_count"] for i in train]),np.log1p(tr["api_total"])]}
        xva={"H0_ROLE":np.c_[va["role"],np.zeros((len(val),API_DIM))],
          "H1_ROLE_API":np.c_[va["role"],va["api"]],"H1_ROLE_API_RAW":np.c_[va["role"],va_raw["api"]],
          "COUNT_ONLY":np.c_[va["host_count"],va["api_total"],np.asarray([audits[i]["unique_api_count"] for i in val]),np.log1p(va["api_total"])]}
        for name in METHODS:
            x=np.r_[xtr[name],xva[name]]
            local_train=np.arange(len(train));local_val=np.arange(len(train),len(train)+len(val))
            oof[name][val]=fit_probe(x,np.r_[y[train],y[val]],local_train,local_val,17000+f)
            feature_norm[name][val]=np.linalg.norm(xva[name],axis=1)
        unknown_rate[val]=va["unknown_rate"]
        vocab_path=ROOT/f"vocabulary_fold_{f}.json"
        vocab_path.write_text(json.dumps({"fold":f,"train_rows":len(train),"validation_rows":len(val),
          "min_apk_df":2,"max_vocab":2048,"vocabulary_size":len(projector.vocabulary),
          "svd_components":projector.components,"vocabulary_sha256":projector.vocab_hash,
          "tokens":[{"api_identity_hash":api,"train_apk_df":projector.frequency[api]} for api in sorted(projector.vocabulary,key=projector.vocabulary.get)]},indent=2)+"\n",encoding="utf-8")
        vocab_audit.append({"fold":f,"train_rows":len(train),"validation_rows":len(val),"vocabulary_size":len(projector.vocabulary),
          "svd_components":projector.components,"vocabulary_sha256":projector.vocab_hash,
          "validation_unknown_rate":float(np.mean(va["unknown_rate"])),"validation_api_occurrences":int(va["api_total"].sum()),
          "validation_unknown_occurrences":int(va["unknown"].sum())})
        host_cursor=0
        for local,apk_i in enumerate(val):
            status="UNROUTED"
            base_ok=(base[apk_i]>=.5)==y[apk_i];final_ok=(final[apk_i]>=.5)==y[apk_i]
            truth=1 if y[apk_i]==1 else -1
            aligned=current_delta[apk_i]*truth
            if route[apk_i]:
                if not base_ok:status="FIXED" if final_ok else ("RIGHT_DIRECTION_TOO_WEAK" if aligned>0 else "WRONG_OR_ZERO_DIRECTION")
                else:status="BROKEN" if not final_ok else "PRESERVED"
            for h,host in enumerate(bags[apk_i]):
                emb=va["host_api"][host_cursor]
                if route[apk_i]:
                    host_records.append({"sha256":ids[apk_i],"fold":f,"host_index":h,"host_node_index":host["host_node_index"],
                      "host_node_id":host["host_node_id"],"label":int(y[apk_i]),"stage1_prob":float(base[apk_i]),
                      "stage1_pred":int(base[apk_i]>=.5),"stage1_error":int(not base_ok),"route":1,
                      "error_type":"FN" if (not base_ok and y[apk_i]==1) else ("FP" if not base_ok else "NONE"),
                      "truth_sign":truth,"current_delta":float(current_delta[apk_i]),"outcome":status,
                      "role_ids":list(host["role_ids"]),"api_ids":list(host["api_ids"]),"api_count":len(host["api_ids"]),
                      "api_embedding":emb.tolist(),"api_unknown_count":sum(api not in projector.vocabulary for api in host["api_ids"]),
                      "c43":c43[apk_i][h].tolist()})
                host_cursor+=1
        if host_cursor!=len(va["host_api"]):raise RuntimeError("Host embedding order mismatch")
        print(f"API identity outer fold {f}/5 done vocab={len(projector.vocabulary)} unk={np.mean(va['unknown_rate']):.6f}",flush=True)
    for name in METHODS:
        if not np.isfinite(oof[name]).all():raise RuntimeError("incomplete OOF probe: "+name)
    direction=[direction_row("E_CURRENT_C43_DELTA",y,base,route,current_score,fold)]
    direction += [direction_row(name,y,base,route,oof[name],fold) for name in METHODS]
    error=route&((base>=.5)!=y);per_fold=[]
    for f in range(1,6):
        s=error&(fold==f);row={"Fold":f,"RoutedErrors":int(s.sum()),"FN":int(np.sum(s&(y==1))),"FP":int(np.sum(s&(y==0)))}
        for name,score in [("E_CURRENT_C43_DELTA",current_score),*[(n,oof[n]) for n in METHODS]]:
            row[name+"_acc"]=float(np.mean((score[s]>=.5)==y[s]))
        per_fold.append(row)
    subgroup=[]
    predicted=base>=.5
    for comparison,subset in (("FP_vs_TP",route&predicted),("FN_vs_TN",route&~predicted)):
        for name in METHODS:
            subgroup.append({"Comparison":comparison,"Representation":name,**metric(y[subset],oof[name][subset])})
    paired=[exact_paired(y,base,route,oof["H1_ROLE_API"],oof["H0_ROLE"],"H1_ROLE_API_minus_H0_ROLE")]
    for label,mask in (("FN",error&(y==1)),("FP",error&(y==0))):
        a=(oof["H1_ROLE_API"][mask]>=.5)==y[mask];b=(oof["H0_ROLE"][mask]>=.5)==y[mask];n01=int(np.sum(a&~b));n10=int(np.sum(~a&b));n=n01+n10
        paired.append({"comparison":f"H1_minus_H0_{label}","candidate_only_correct":n01,"control_only_correct":n10,
          "paired_net":n01-n10,"exact_p":float(binomtest(min(n01,n10),n,.5).pvalue) if n else 1.})
    extra,extra_schema=extra_size_counts(ids);count_names=("host_count","api_occurrence_count","unique_api_count","graph_node_count","graph_edge_count","apk_method_count","audit_graph_node_count","audit_graph_edge_count")
    count_values={name:np.asarray([float(extra[sha].get(name,audits[i].get(name,np.nan))) for i,sha in enumerate(ids)]) for name in count_names}
    shortcut=[]
    for name in METHODS:
        for count_name,values in count_values.items():
            valid=np.isfinite(values)
            rho_norm=float(spearmanr(feature_norm[name][valid],values[valid]).statistic) if valid.sum()>2 else float("nan")
            rho_score=float(spearmanr(oof[name][valid],values[valid]).statistic) if valid.sum()>2 else float("nan")
            shortcut.append({"Representation":name,"AuditVariable":count_name,"n":int(valid.sum()),
              "corr_feature_norm":rho_norm,"corr_score":rho_score})
    nn_rows,collapse=collapse_audit(host_records)
    write_host_audit(ROOT/"routed_host_audit.jsonl.gz",host_records)
    write_csv(ROOT/"nearest_neighbor_collapse.csv",nn_rows)
    write_csv(ROOT/"direction_quality.csv",direction);write_csv(ROOT/"per_fold_direction.csv",per_fold)
    write_csv(ROOT/"failure_taxonomy.csv",[{"Representation":r["Representation"],"WrongDirection":r["WrongDirection"],"Ambiguous":r["Ambiguous"],"CorrectDirection":r["CorrectDirection"]} for r in direction])
    write_csv(ROOT/"fp_tp_fn_tn_metrics.csv",subgroup);write_csv(ROOT/"shortcut_audit.csv",shortcut)
    write_csv(ROOT/"vocabulary_audit.csv",vocab_audit);write_csv(ROOT/"paired_direction.csv",paired)
    write_csv(ROOT/"api_reconstruction_failures.csv",failures)
    prediction=[]
    for i,sha in enumerate(ids):
        row={"sha256":sha,"fold":int(fold[i]),"label":int(y[i]),"stage1_prob":float(base[i]),"route":int(route[i]),
             "current_c43_prob":float(final[i]),"current_delta":float(current_delta[i]),"unknown_rate":float(unknown_rate[i]),**audits[i]}
        for name in METHODS:row[name+"__prob"]=float(oof[name][i]);row[name+"__feature_norm"]=float(feature_norm[name][i])
        prediction.append(row)
    write_csv(ROOT/"source_oof_predictions.csv",prediction)
    coverage={"source_rows":799,"routed_rows":int(route.sum()),"routed_errors":int(error.sum()),"FN":int(np.sum(error&(y==1))),"FP":int(np.sum(error&(y==0))),
      "total_hosts":int(sum(a["host_count"] for a in audits)),"total_api_occurrences":int(sum(a["api_occurrence_count"] for a in audits)),
      "unique_api_identity_hashes":len(set(api for bag in bags for host in bag for api in host["api_ids"])),
      "mean_unknown_rate":float(np.mean(unknown_rate)),"endpoint_cache_load_seconds":load_seconds,
      "host_api_reconstruction_seconds":reconstruction_seconds,"mean_reconstruction_seconds_per_apk":reconstruction_seconds/799,
      "failure_count":len(failures),"pair_coverage":"NOT_RUN_E1_GATE_REQUIRED","path_coverage":"NOT_RUN_E1_GATE_REQUIRED"}
    (ROOT/"coverage_cost.json").write_text(json.dumps(coverage,indent=2)+"\n",encoding="utf-8")
    (ROOT/"representation_collapse.json").write_text(json.dumps(collapse,indent=2)+"\n",encoding="utf-8")
    schema={"H0_ROLE":{"width":62,"active":"Role31","padding":"31 zeros for capacity-aligned input"},
      "H1_ROLE_API":{"width":62,"active":"Role31 + train-only normalized API-SVD31"},
      "H1_ROLE_API_RAW":{"width":62,"active":"Role31 + sum-pooled API-SVD31; shortcut diagnostic only"},
      "COUNT_ONLY":{"width":4,"active":"host count, API occurrence count, unique API count, log1p API occurrence"},
      "host_identity_source":"endpoint payload node_ids (stable canonical-signature hashes)","compact43_source":"first 43 columns of frozen 105-D sidecar",
      "compact_offsets_key":offsets_key,"compact_values_key":values_key,"extra_size_schema":extra_schema}
    (ROOT/"feature_schema.json").write_text(json.dumps(schema,indent=2)+"\n",encoding="utf-8")
    h0=next(r for r in direction if r["Representation"]=="H0_ROLE");h1=next(r for r in direction if r["Representation"]=="H1_ROLE_API")
    gains=[]
    for frow in per_fold:gains.append(frow["H1_ROLE_API_acc"]-frow["H0_ROLE_acc"])
    advance=(h1["FPDirectionAcc"]-h0["FPDirectionAcc"]>=.10 and h1["FNDirectionAcc"]>=h0["FNDirectionAcc"]-.05 and h1["OverallDirectionAcc"]>h0["OverallDirectionAcc"] and sum(g>=0 for g in gains)>=3)
    decision={"advance_api_identity_to_c43":bool(advance),"gate":{"min_FP_gain":.10,"max_FN_drop":.05,"overall_gain_required":True,"nonnegative_folds_required":3},
      "observed":{"FP_gain":h1["FPDirectionAcc"]-h0["FPDirectionAcc"],"FN_change":h1["FNDirectionAcc"]-h0["FNDirectionAcc"],
        "overall_gain":h1["OverallDirectionAcc"]-h0["OverallDirectionAcc"],"nonnegative_folds":sum(g>=0 for g in gains)},
      "next_step":"E2_PAIR_ALLOWED" if advance else "STOP_AFTER_E1","future_diagnostic_authorized":False}
    (ROOT/"advance_decision.json").write_text(json.dumps(decision,indent=2)+"\n",encoding="utf-8")
    print("P31RF_C43_APIIDENTITY_PROBE_DONE")
    print(f"routed={route.sum()} errors={error.sum()} FN={np.sum(error&(y==1))} FP={np.sum(error&(y==0))}")
    for row in direction:print(f"{row['Representation']}: overall={row['OverallDirectionAcc']:.6f} FN={row['FNDirectionAcc']:.6f} FP={row['FPDirectionAcc']:.6f} wrong={row['WrongDirection']} ambiguous={row['Ambiguous']}")
    print("E1_ADVANCE_DECISION="+("PASS" if advance else "STOP"))

if __name__=="__main__":main()

