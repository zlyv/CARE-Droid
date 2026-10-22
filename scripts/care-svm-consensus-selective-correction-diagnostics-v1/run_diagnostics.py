#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from common import *

def load_stage2(path):
    rows=[row for row in read_csv(path) if norm(row.get("arm"))=="svm"]
    if len(rows)!=1989: raise RuntimeError(f"BLOCKED_BY_CONTRACT: SVM diagnostic rows changed path={path} rows={len(rows)}")
    return rows


def frozen_member_map():
    result={};svm={}
    for row in read_csv(FROZEN_MEMBERS):
        sid=norm(row["sha256"]);result[sid]=np.asarray([float(row["p_P31__LR"]),float(row["p_P31__RF"]),float(row["p_P31__XGB"])])
        svm[sid]=float(row["p_P31__SVM"])
    if len(result)!=1989: raise RuntimeError("BLOCKED_BY_CONTRACT: frozen member cohort changed")
    return result,svm


def within_member_map():
    rows=read_csv(WITHIN_MEMBERS); grouped={}
    for row in rows:
        key=norm(row["sha256"]);arm=str(row["arm"]).strip().upper()
        if arm in {"LR","RF","XGB","SVM"}: grouped.setdefault(key,{})[arm]=float(row["probability"])
    result={sid:np.asarray([value["LR"],value["RF"],value["XGB"]]) for sid,value in grouped.items() if {"LR","RF","XGB","SVM"}<=set(value)}
    svm={sid:value["SVM"] for sid,value in grouped.items() if "SVM" in value}
    if len(result)!=1989: raise RuntimeError(f"BLOCKED_BY_CONTRACT: within member cohort changed {len(result)}")
    return result,svm


def evaluate(protocol,rows,member_map,svm_map=None):
    rows=sorted(rows,key=lambda r:norm(r["sha256"]));sha=np.asarray([norm(r["sha256"]) for r in rows])
    year=np.asarray([int(r["year"]) for r in rows]);fold=np.asarray([int(r.get("fold",0) or 0) for r in rows])
    y=np.asarray([int(r["label"]) for r in rows]);base=np.asarray([float(r["stage1_prob"]) for r in rows])
    cand=np.asarray([float(r["stage2_prob"]) for r in rows]);route=np.asarray([int(r["routed"]) for r in rows],bool)
    other=np.asarray([member_map[sid] for sid in sha]);candidate_label=cand>=.5;base_label=base>=.5;flip=candidate_label!=base_label
    if svm_map is not None:
        error=float(np.max(np.abs(base-np.asarray([svm_map[sid] for sid in sha]))))
        if error>1e-10: raise RuntimeError(f"BLOCKED_BY_CONTRACT: same-year SVM alignment error {error}")
    support=np.sum((other>=.5)==candidate_label[:,None],axis=1)
    majority=np.where(flip&(support<2),base,cand);unanimous=np.where(flip&(support<3),base,cand)
    finals={"S0_STAGE1_SVM":base,"S1_ALL_C43":cand,"S2_MAJORITY_2_OF_3":majority,"S3_UNANIMOUS_3_OF_3":unanimous}
    aggregate=[];per_fold=[];predictions=[]
    scopes=[(str(v),year==v) for v in YEARS]+[("pooled",np.ones(len(y),bool))]
    for scope,selected in scopes:
        for arm in ARMS:
            aggregate.append({"protocol":protocol,"scope":scope,"arm":arm,**metrics(y[selected],base[selected],finals[arm][selected],route[selected])})
    if protocol=="independent_same_year_fixed_5fold":
        for current_year in YEARS:
            for current_fold in range(1,6):
                selected=(year==current_year)&(fold==current_fold)
                for arm in ARMS:
                    per_fold.append({"protocol":protocol,"year":current_year,"fold":current_fold,"arm":arm,
                        **metrics(y[selected],base[selected],finals[arm][selected],route[selected])})
    for i,sid in enumerate(sha):
        for arm in ARMS:
            predictions.append({"protocol":protocol,"sha256":sid,"year":int(year[i]),"fold":int(fold[i]) if fold[i] else "",
                "label":int(y[i]),"arm":arm,"stage1_prob":float(base[i]),"candidate_prob":float(cand[i]),
                "final_prob":float(finals[arm][i]),"routed":int(route[i]),"candidate_flip":int(flip[i]),
                "other_support_count":int(support[i])})
    return aggregate,per_fold,predictions


def matched_random(protocol,rows,member_map,iterations=2000):
    rows=sorted(rows,key=lambda r:norm(r["sha256"]));year=np.asarray([int(r["year"]) for r in rows]);y=np.asarray([int(r["label"]) for r in rows])
    base=np.asarray([float(r["stage1_prob"]) for r in rows]);cand=np.asarray([float(r["stage2_prob"]) for r in rows]);route=np.asarray([int(r["routed"]) for r in rows],bool)
    label=cand>=.5;flip=(base>=.5)!=label;support=np.asarray([np.sum((member_map[norm(r["sha256"])]>=.5)==label[i]) for i,r in enumerate(rows)])
    accept=support>=2;majority=np.where(flip&(~accept),base,cand);observed=metrics(y,base,majority,route)
    rng=np.random.RandomState(SEED+(0 if protocol.startswith("frozen") else 1));values=[]
    for iteration in range(iterations):
        chosen=np.zeros(len(y),bool)
        for current_year in YEARS:
            for proposed in (0,1):
                eligible=np.flatnonzero(flip&(year==current_year)&(label==proposed));count=int(np.sum(accept[eligible]))
                if count: chosen[rng.choice(eligible,count,replace=False)]=True
        final=np.where(flip&(~chosen),base,cand);m=metrics(y,base,final,route);values.append(m["net"])
    values=np.asarray(values,float)
    return {"protocol":protocol,"iterations":iterations,"majority_net":observed["net"],"random_net_mean":float(values.mean()),
        "random_net_95pct":float(np.quantile(values,.95)),"one_sided_p_net":float((1+np.sum(values>=observed["net"]))/(1+len(values)))}


def main():
    frozen=load_stage2(FROZEN_STAGE2);within=load_stage2(WITHIN_STAGE2);fm,fsvm=frozen_member_map();wm,wsvm=within_member_map()
    fa,ff,fp=evaluate("frozen_2017_to_future",frozen,fm,fsvm);wa,wf,wp=evaluate("independent_same_year_fixed_5fold",within,wm,wsvm)
    aggregate=fa+wa;write_csv(OUTPUT/"diagnostic_results.csv",aggregate);write_csv(OUTPUT/"same_year_fold_results.csv",wf)
    write_csv(OUTPUT/"diagnostic_predictions.csv",fp+wp)
    controls=[matched_random("frozen_2017_to_future",frozen,fm),matched_random("independent_same_year_fixed_5fold",within,wm)]
    dump(OUTPUT/"matched_random_controls.json",controls)
    summaries=[]
    for protocol in ("frozen_2017_to_future","independent_same_year_fixed_5fold"):
        for scope in [str(y) for y in YEARS]+["pooled"]:
            block={row["arm"]:row for row in aggregate if row["protocol"]==protocol and row["scope"]==scope}
            summaries.append({"protocol":protocol,"scope":scope,"all_c43_net":block["S1_ALL_C43"]["net"],
                "majority_net":block["S2_MAJORITY_2_OF_3"]["net"],"unanimous_net":block["S3_UNANIMOUS_3_OF_3"]["net"],
                "majority_net_change":block["S2_MAJORITY_2_OF_3"]["net"]-block["S1_ALL_C43"]["net"],
                "majority_BA_change_pp":100*(block["S2_MAJORITY_2_OF_3"]["BA"]-block["S1_ALL_C43"]["BA"]),
                "majority_fixed":block["S2_MAJORITY_2_OF_3"]["fixed"],"majority_broken":block["S2_MAJORITY_2_OF_3"]["broken"]})
    write_csv(OUTPUT/"diagnostic_comparison.csv",summaries)
    dump(OUTPUT/"diagnostic_decision.json",{"status":"DIAGNOSTIC_COMPLETE_SOURCE_DECISION_UNCHANGED",
        "source_decision":"SVM_CONSENSUS_GATE_NOT_SUPPORTED","future_used_for_selection":False,
        "thresholds_changed_after_future":False,"matched_random_controls":controls,
        "interpretation":"Future may reveal behavior but cannot rescue the failed 2017 source gate."})
    print("SVM_CONSENSUS_FUTURE_AND_WITHIN_DIAGNOSTICS_DONE")


if __name__=="__main__":main()
