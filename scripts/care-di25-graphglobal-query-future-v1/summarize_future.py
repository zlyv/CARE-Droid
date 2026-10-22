#!/usr/bin/env python3
"""Open future labels only after both frozen label-free predictions exist."""
from __future__ import annotations
import json, math, sys
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from common import FUTURE_INPUT, RESOLVED, ROOT, ROUTE_COUNTS, YEARS, YEAR_COUNTS, norm, read_csv, sha256, unique, write_csv

ARMS=("N0_NO_GRAPHGLOBAL","Q_QUERY_ONLY")
METHOD={
    "N0_NO_GRAPHGLOBAL":"DI25__Host__Anomaly__N0_NoGraphGlobal__Bounded",
    "Q_QUERY_ONLY":"DI25__Host__Anomaly__Q_GraphGlobalQueryOnly__Bounded",
}

def parse_label(value):
    text=norm(value)
    if text in {"1","1.0","malware","malicious","true"}: return 1
    if text in {"0","0.0","benign","goodware","false"}: return 0
    raise RuntimeError("unknown post-hoc label "+repr(value))

def metric(y,p,base,route,method,scope,year=""):
    y=np.asarray(y,int);p=np.asarray(p,float);base=np.asarray(base,float);route=np.asarray(route,bool)
    pred=p>=.5;base_ok=(base>=.5)==y;ok=pred==y
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    fixed=int(np.sum(route & ~base_ok & ok));broken=int(np.sum(route & base_ok & ~ok))
    return {"scope":scope,"year":year,"method":method,"n":len(y),"routed":int(route.sum()),
        "routing_rate":float(route.mean()),"BA":float(balanced_accuracy_score(y,pred)),
        "AUC":float(roc_auc_score(y,p)),"F1":float(f1_score(y,pred,zero_division=0)),
        "TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp),"fixed":fixed,"broken":broken,
        "net":fixed-broken,"correction_precision":fixed/(fixed+broken) if fixed+broken else float("nan"),
        "diagnostic_only":1}

def exact_p(n01,n10):
    n=n01+n10
    if n==0:return 1.0
    k=min(n01,n10)
    return min(1.0,2.0*sum(math.comb(n,i) for i in range(k+1))/(2**n))

def main():
    # Freeze and hash both label-free outputs before the label path is opened.
    predictions={};pred_hash={};protocols={}
    for arm in ARMS:
        path=ROOT/"arms"/arm/"future_predictions_label_free.csv"
        protocol_path=ROOT/"arms"/arm/"protocol.json"
        protocols[arm]=json.loads(protocol_path.read_text(encoding="utf-8"))
        if not protocols[arm].get("source_only_training") or protocols[arm].get("future_labels_loaded_by_model") or protocols[arm].get("ego_consumed"):
            raise RuntimeError("BLOCKED_BY_CONTRACT: invalid label-free protocol "+arm)
        predictions[arm]=unique(read_csv(path),1989);pred_hash[arm]=sha256(path)
    frozen=read_csv(FUTURE_INPUT);index=unique(frozen,1989)
    resolved=json.loads(RESOLVED.read_text(encoding="utf-8"))
    label_path=Path(resolved["future_label_file_posthoc_only"])
    labeled=read_csv(label_path)
    label_by_sha={norm(r.get("sha256") or r.get("sample_id")):parse_label(r.get("label",r.get("metadata_label"))) for r in labeled}
    if len(label_by_sha)!=1989 or set(label_by_sha)!=set(index):
        raise RuntimeError("post-hoc label cohort changed")
    ids=[norm(r["sha256"]) for r in frozen]
    years=np.asarray([int(float(r["year"])) for r in frozen],int)
    y=np.asarray([label_by_sha[s] for s in ids],int)
    base=np.asarray([float(index[s]["p_stage1"]) for s in ids],float)
    route=np.asarray([int(float(index[s]["route_DI"]))==1 for s in ids],bool)
    expert={arm:np.asarray([float(predictions[arm][s]["p_expert"]) for s in ids]) for arm in ARMS}
    if not all(np.isfinite(v).all() for v in (base,*expert.values())): raise RuntimeError("non-finite probability")
    final={"Global24":base}
    final.update({METHOD[a]:np.where(route,expert[a],base) for a in ARMS})
    metrics=[]
    for method,p in final.items():
        used=np.zeros(len(route),bool) if method=="Global24" else route
        for year in YEARS:
            take=years==year; metrics.append(metric(y[take],p[take],base[take],used[take],method,"annual",year))
        metrics.append(metric(y,p,base,used,method,"pooled_2018_2021"))
    paired=[]
    n0=final[METHOD["N0_NO_GRAPHGLOBAL"]];q=final[METHOD["Q_QUERY_ONLY"]]
    for scope,take in [(str(year),years==year) for year in YEARS]+[("pooled_2018_2021",np.ones(len(y),bool))]:
        c0=(n0[take]>=.5)==y[take];cq=(q[take]>=.5)==y[take]
        n01=int(np.sum(~c0&cq));n10=int(np.sum(c0&~cq))
        paired.append({"scope":scope,"comparison":"Q_QUERY_ONLY_minus_N0_NO_GRAPHGLOBAL",
            "n01_N0_wrong_Q_correct":n01,"n10_N0_correct_Q_wrong":n10,"paired_net_Q_minus_N0":n01-n10,
            "exact_McNemar_two_sided_p_value":exact_p(n01,n10)})
    samples=[]
    for i,s in enumerate(ids):
        samples.append({"sha256":s,"year":int(years[i]),"label":int(y[i]),"p_stage1":float(base[i]),
            "route_DI":int(route[i]),"p_expert_N0":float(expert["N0_NO_GRAPHGLOBAL"][i]),
            "p_expert_Q":float(expert["Q_QUERY_ONLY"][i]),"p_final_N0":float(n0[i]),"p_final_Q":float(q[i]),
            "N0_correct":int((n0[i]>=.5)==y[i]),"Q_correct":int((q[i]>=.5)==y[i])})
    out=ROOT/"results";out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"future_metrics.csv",metrics);write_csv(out/"future_N0_Q_paired.csv",paired);write_csv(out/"future_sample_predictions.csv",samples)
    pooled=[r for r in metrics if r["scope"]=="pooled_2018_2021"]
    annual=[r for r in metrics if r["scope"]=="annual"]
    lines=["CARE-Droid frozen GraphGlobal query-only future diagnostic v1","",
        "Q_QUERY_ONLY was frozen by human approval using 2017 source evidence before this run.",
        "N0_NO_GRAPHGLOBAL is a mechanism control, not a future-selected alternative.",
        "Both arms were fitted on the fixed 799-APK 2017 cohort only; no Ego tensor was consumed.",
        "Future labels were opened only by this summarizer after label-free prediction files were frozen.",
        "2018-2021 were previously inspected and are diagnostic-only; these results must not revise the method.","","Pooled 2018-2021:"]
    for r in pooled: lines.append("{method}: BA={BA:.6f} AUC={AUC:.6f} F1={F1:.6f} FP={FP} FN={FN} fixed={fixed} broken={broken} net={net}".format(**r))
    lines += ["","Annual:"]
    for year in YEARS:
        lines.append(str(year))
        for r in annual:
            if int(r["year"])==year: lines.append("  {method}: routed={routed}/{n} BA={BA:.6f} AUC={AUC:.6f} F1={F1:.6f} fixed={fixed} broken={broken} net={net}".format(**r))
    lines += ["","Paired Q versus N0:"]
    for r in paired: lines.append("  {scope}: n01={n01_N0_wrong_Q_correct} n10={n10_N0_correct_Q_wrong} net={paired_net_Q_minus_N0} p={exact_McNemar_two_sided_p_value:.6f}".format(**r))
    (out/"REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    audit={"prediction_sha256_before_label_unseal":pred_hash,"future_labels_used_by_model":False,
        "future_labels_unsealed_only_by":"summarize_future.py","future_diagnostic_only":True,
        "year_counts":dict(Counter(int(x) for x in years)),
        "route_counts":{str(year):int(np.sum(route&(years==year))) for year in YEARS},
        "method_revision_authorized":False}
    (out/"unseal_audit.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    print("\n".join(lines),flush=True)
    print("GRAPHGLOBAL_QUERY_FUTURE_SUMMARY_PASS",flush=True)

if __name__=="__main__": main()
