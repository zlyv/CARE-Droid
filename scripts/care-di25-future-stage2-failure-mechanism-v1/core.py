#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np

ROOT=Path("artifacts/care-di25-graphglobal-query-future-v1")
SAMPLES=ROOT/"results/future_sample_predictions.csv"
Q_FREE=ROOT/"arms/Q_QUERY_ONLY/future_predictions_label_free.csv"
N0_FREE=ROOT/"arms/N0_NO_GRAPHGLOBAL/future_predictions_label_free.csv"
Q_PROTOCOL=ROOT/"arms/Q_QUERY_ONLY/protocol.json"
N0_PROTOCOL=ROOT/"arms/N0_NO_GRAPHGLOBAL/protocol.json"
ROUTER=Path("artifacts/care-route-stage2-future-matrix-v1/preflight/future_router_inputs_label_free.csv")
EXPECTED={2018:793,2019:399,2020:397,2021:400};ROUTES={2018:198,2019:100,2020:99,2021:100}

def read(p):
    with Path(p).open("r",encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def norm(v):return str(v or "").strip().lower()
def idx(rows):
    out={}
    for r in rows:
        k=norm(r.get("sha256") or r.get("sample_id"))
        if not k or k in out:raise RuntimeError("duplicate/missing SHA "+k)
        out[k]=r
    return out
def label(v):
    x=norm(v)
    if x in ("1","1.0","malware","malicious","true"):return 1
    if x in ("0","0.0","benign","goodware","false"):return 0
    raise RuntimeError("unknown label "+repr(v))
def logit(p):
    x=np.clip(np.asarray(p,float),1e-8,1-1e-8);return np.log(x/(1-x))
def write(path,rows):
    if not rows:raise RuntimeError("empty output "+str(path))
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)

def validate_protocol(path,arm):
    p=json.loads(path.read_text(encoding="utf-8"))
    expected={"arm":arm,"source_only_training":True,"future_labels_loaded_by_model":False,
              "router_applied_during_model_fit":False,"ego_consumed":False,
              "parameter_count":33036,"max_delta_logit":2.25}
    bad={k:(p.get(k),v) for k,v in expected.items() if p.get(k)!=v}
    if bad:raise RuntimeError("BLOCKED_BY_CONTRACT: protocol changed "+repr(bad))
    return p

def load():
    for p in (SAMPLES,Q_FREE,N0_FREE,Q_PROTOCOL,N0_PROTOCOL,ROUTER):
        if not p.is_file() or p.stat().st_size<100:raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty "+str(p))
    validate_protocol(Q_PROTOCOL,"Q_QUERY_ONLY");validate_protocol(N0_PROTOCOL,"N0_NO_GRAPHGLOBAL")
    s=idx(read(SAMPLES));q=idx(read(Q_FREE));n0=idx(read(N0_FREE));r=idx(read(ROUTER))
    if len(s)!=1989 or set(s)!=set(q) or set(s)!=set(n0) or set(s)!=set(r):raise RuntimeError("cohort mismatch")
    ids=[norm(x["sha256"]) for x in read(ROUTER)]
    years=np.asarray([int(float(r[x]["year"])) for x in ids]);y=np.asarray([label(s[x]["label"]) for x in ids])
    base=np.asarray([float(r[x]["p_stage1"]) for x in ids]);route=np.asarray([int(float(r[x]["route_DI"]))==1 for x in ids])
    qp=np.asarray([float(q[x]["p_expert"]) for x in ids]);np0=np.asarray([float(n0[x]["p_expert"]) for x in ids])
    hosts=np.asarray([int(float(q[x].get("real_host_count",0))) for x in ids])
    member=np.column_stack([[float(r[x][k]) for x in ids] for k in ("p_lr","p_rf","p_xgb","p_svm")])
    if {z:int((years==z).sum()) for z in EXPECTED}!=EXPECTED or {z:int((route&(years==z)).sum()) for z in ROUTES}!=ROUTES:raise RuntimeError("cohort/route changed")
    exposed=np.asarray([float(s[x]["p_expert_Q"]) for x in ids]);exposed0=np.asarray([float(s[x]["p_expert_N0"]) for x in ids])
    if np.max(np.abs(qp-exposed))>1e-12 or np.max(np.abs(np0-exposed0))>1e-12:raise RuntimeError("saved expert mismatch")
    return ids,years,y,base,route,qp,np0,hosts,member

def failure_type(y,base,q,max_delta=2.25,near=.05):
    lb=float(logit([base])[0]);lq=float(logit([q])[0]);delta=lq-lb
    desired=1.0 if int(y)==1 else -1.0;aligned=desired*delta;required=abs(lb)+1e-6
    if (q>=.5)==bool(y):kind="FIXED"
    elif abs(delta)<near:kind="NEAR_ZERO"
    elif aligned<=0:kind="WRONG_DIRECTION"
    elif required>max_delta:kind="BOUND_LIMIT"
    else:kind="RIGHT_DIRECTION_TOO_WEAK"
    return kind,lb,lq,delta,desired,aligned,required

