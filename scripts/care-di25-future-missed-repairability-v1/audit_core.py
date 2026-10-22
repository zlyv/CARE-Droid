#!/usr/bin/env python3
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

ROOT=Path("artifacts/care-di25-graphglobal-query-future-v1")
SAMPLES=ROOT/"results/future_sample_predictions.csv"
LABEL_FREE=ROOT/"arms/Q_QUERY_ONLY/future_predictions_label_free.csv"
PROTOCOL=ROOT/"arms/Q_QUERY_ONLY/protocol.json"
ROUTER=Path("artifacts/care-route-stage2-future-matrix-v1/preflight/future_router_inputs_label_free.csv")
EXPECTED={2018:793,2019:399,2020:397,2021:400}
ROUTES={2018:198,2019:100,2020:99,2021:100}

def read(path):
    with Path(path).open("r",encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def norm(v):return str(v or "").strip().lower()
def index(rows):
    out={}
    for r in rows:
        k=norm(r.get("sha256") or r.get("sample_id"))
        if not k or k in out:raise RuntimeError("missing/duplicate SHA "+k)
        out[k]=r
    return out
def label(v):
    x=norm(v)
    if x in ("1","1.0","malware","malicious","true"):return 1
    if x in ("0","0.0","benign","goodware","false"):return 0
    raise RuntimeError("unknown label "+repr(v))
def write(path,rows):
    if not rows:raise RuntimeError("empty output "+str(path))
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)

def load_and_validate():
    for p in (SAMPLES,LABEL_FREE,PROTOCOL,ROUTER):
        if not p.is_file() or p.stat().st_size<100:raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty "+str(p))
    protocol=json.loads(PROTOCOL.read_text(encoding="utf-8"))
    required={"source_only_training":True,"future_labels_loaded_by_model":False,"router_applied_during_model_fit":False,
              "arm":"Q_QUERY_ONLY","max_delta_logit":2.25,"parameter_count":33036,"ego_consumed":False}
    bad={k:(protocol.get(k),v) for k,v in required.items() if protocol.get(k)!=v}
    if bad:raise RuntimeError("BLOCKED_BY_CONTRACT: Q protocol changed "+repr(bad))
    samples=read(SAMPLES);free=read(LABEL_FREE);router=read(ROUTER)
    si=index(samples);fi=index(free);ri=index(router)
    if len(si)!=1989 or set(si)!=set(fi) or set(si)!=set(ri):raise RuntimeError("BLOCKED_BY_CONTRACT: cohorts do not align")
    ids=[norm(r["sha256"]) for r in router]
    years=np.asarray([int(float(ri[s]["year"])) for s in ids],int)
    if {y:int((years==y).sum()) for y in EXPECTED}!=EXPECTED:raise RuntimeError("year cohort changed")
    y=np.asarray([label(si[s]["label"]) for s in ids],int)
    base=np.asarray([float(ri[s]["p_stage1"]) for s in ids],float)
    route=np.asarray([int(float(ri[s]["route_DI"]))==1 for s in ids],bool)
    expert=np.asarray([float(fi[s]["p_expert"]) for s in ids],float)
    exposed=np.asarray([float(si[s]["p_expert_Q"]) for s in ids],float)
    final=np.asarray([float(si[s]["p_final_Q"]) for s in ids],float)
    if {yr:int((route&(years==yr)).sum()) for yr in ROUTES}!=ROUTES:raise RuntimeError("frozen route changed")
    max_expert=float(np.max(np.abs(expert-exposed)))
    expected_final=np.where(route,expert,base);max_final=float(np.max(np.abs(expected_final-final)))
    if max_expert>1e-12 or max_final>1e-12:raise RuntimeError("BLOCKED_BY_CONTRACT: saved probabilities do not reproduce")
    return ids,years,y,base,route,expert,{"p_expert_max_abs_diff":max_expert,"p_final_max_abs_diff":max_final,"protocol":protocol}

