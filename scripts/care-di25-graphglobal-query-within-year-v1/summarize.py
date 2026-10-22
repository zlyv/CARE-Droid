#!/usr/bin/env python3
"""Summarize N0/Q same-year OOF; this is not temporal generalization."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score,confusion_matrix,f1_score,roc_auc_score

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from common import ARMS,ROOT,ROUTER,YEARS,exact_p,norm,read_csv,unique,write_csv

METHOD={"N0_NO_GRAPHGLOBAL":"DI25__Host__Anomaly__N0_NoGraphGlobal__Bounded","Q_QUERY_ONLY":"DI25__Host__Anomaly__Q_GraphGlobalQueryOnly__Bounded"}

def safe_auc(y,p):
    return float(roc_auc_score(y,p)) if len(np.unique(y))==2 else float("nan")

def metric(y,p,base,route,method,scope,year="",fold=""):
    y=np.asarray(y,int);p=np.asarray(p,float);base=np.asarray(base,float);route=np.asarray(route,bool)
    pred=p>=.5;base_ok=(base>=.5)==y;ok=pred==y;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    fixed=int(np.sum(route&~base_ok&ok));broken=int(np.sum(route&base_ok&~ok))
    return {"scope":scope,"year":year,"fold":fold,"method":method,"n":len(y),"routed":int(route.sum()),
        "BA":float(balanced_accuracy_score(y,pred)),"AUC":safe_auc(y,p),"F1":float(f1_score(y,pred,zero_division=0)),
        "TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp),"fixed":fixed,"broken":broken,"net":fixed-broken}

def main():
    router_rows=read_csv(ROUTER);router=unique(router_rows)
    ids=[norm(r["sha256"]) for r in router_rows];years=np.asarray([int(r["year"]) for r in router_rows]);folds=np.asarray([int(r["fold"]) for r in router_rows])
    y=np.asarray([int(r["label"]) for r in router_rows]);base=np.asarray([float(r["p_stage1"]) for r in router_rows]);route=np.asarray([int(float(r["route_DI"]))==1 for r in router_rows])
    expert={}
    for arm in ARMS:
        path=ROOT/"arms"/arm/"within_year_predictions.csv";rows=read_csv(path);index=unique(rows)
        if set(index)!=set(router):raise RuntimeError("arm/router cohort differs "+arm)
        protocol=json.loads((ROOT/"arms"/arm/"protocol.json").read_text(encoding="utf-8"))
        if protocol.get("ego_consumed") or protocol.get("parameter_count")!=33036 or protocol.get("protocol")!="independent same-year five-fold OOF":raise RuntimeError("arm protocol changed "+arm)
        for sha,row in index.items():
            ref=router[sha]
            if int(row["year"])!=int(ref["year"]) or int(row["fold"])!=int(ref["fold"]):raise RuntimeError("fold alignment changed "+sha)
        expert[arm]=np.asarray([float(index[s]["p_expert"]) for s in ids])
    final={"Global24":base};final.update({METHOD[a]:np.where(route,expert[a],base) for a in ARMS})
    annual=[];pooled=[];fold_metrics=[]
    for method,p in final.items():
        used=np.zeros(len(route),bool) if method=="Global24" else route
        pooled.append(metric(y,p,base,used,method,"pooled_2018_2021"))
        for year in YEARS:
            take=years==year;annual.append(metric(y[take],p[take],base[take],used[take],method,"annual",year))
            for fold in range(1,6):
                fmask=take&(folds==fold);fold_metrics.append(metric(y[fmask],p[fmask],base[fmask],used[fmask],method,"fold",year,fold))
    n0=final[METHOD["N0_NO_GRAPHGLOBAL"]];q=final[METHOD["Q_QUERY_ONLY"]];paired=[]
    for scope,take in [(str(year),years==year) for year in YEARS]+[("pooled_2018_2021",np.ones(len(y),bool))]:
        c0=(n0[take]>=.5)==y[take];cq=(q[take]>=.5)==y[take];n01=int(np.sum(~c0&cq));n10=int(np.sum(c0&~cq))
        paired.append({"scope":scope,"n01_N0_wrong_Q_correct":n01,"n10_N0_correct_Q_wrong":n10,"paired_net_Q_minus_N0":n01-n10,"exact_McNemar_two_sided_p_value":exact_p(n01,n10)})
    out=ROOT/"results";out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"annual_metrics.csv",annual);write_csv(out/"pooled_metrics.csv",pooled);write_csv(out/"fold_metrics.csv",fold_metrics);write_csv(out/"paired_Q_vs_N0.csv",paired)
    lines=["CARE-Droid GraphGlobal query-only independent within-year OOF v1","",
        "Each year uses its own strict five-fold OOF Stage-I, DI25 router, prototypes, and Stage-II training.",
        "Host-only; no Ego tensor; N0 and Q have 33,036 parameters and the same bounded head.",
        "This is same-year development/diagnostic evidence, not temporal generalization, unseen-future, or zero-day evidence.","","Pooled:"]
    for r in pooled:lines.append("{method}: BA={BA:.6f} AUC={AUC:.6f} F1={F1:.6f} fixed={fixed} broken={broken} net={net}".format(**r))
    lines += ["","Annual:"]
    for year in YEARS:
        lines.append(str(year))
        for r in annual:
            if int(r["year"])==year:lines.append("  {method}: routed={routed}/{n} BA={BA:.6f} AUC={AUC:.6f} F1={F1:.6f} fixed={fixed} broken={broken} net={net}".format(**r))
    lines += ["","Paired Q versus N0:"]
    for r in paired:lines.append("  {scope}: n01={n01_N0_wrong_Q_correct} n10={n10_N0_correct_Q_wrong} net={paired_net_Q_minus_N0} p={exact_McNemar_two_sided_p_value:.6f}".format(**r))
    (out/"REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (out/"protocol.json").write_text(json.dumps({"same_year_oof":True,"temporal_generalization_result":False,"years":list(YEARS),"arms":list(ARMS),"ego_consumed":False,"parameter_count":33036},indent=2)+"\n",encoding="utf-8")
    print("\n".join(lines));print("GRAPHGLOBAL_QUERY_WITHIN_YEAR_SUMMARY_PASS")

if __name__=="__main__":main()
