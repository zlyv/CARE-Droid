#!/usr/bin/env python3
"""Only this final process unseals future labels."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,f1_score
from safety_core import fixed_broken

PRED=Path("artifacts/care-di25-noego-safe-authority-full-v1/future/future_predictions_label_free.csv")
LINEAGE=Path("artifacts/care-di25-noego-safe-authority-full-v1/preflight/future_lineage_2018_2021_exact.csv")
OUT=PRED.parent
def read(p):
    with p.open("r",encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def norm(x):return str(x).strip().lower()
def label(x):
    value=norm(x)
    if value in {"0","0.0","benign","goodware"}:return 0
    if value in {"1","1.0","malware","malicious"}:return 1
    raise RuntimeError("unknown future label: "+repr(x))
def row_label(row):
    """Accept both the historical and canonical lineage label contracts."""
    for key in ("metadata_label", "label"):
        if str(row.get(key, "")).strip():
            return label(row[key])
    raise RuntimeError("future lineage row has neither metadata_label nor label")
def met(y,p):
    z=p>=.5;return {"BA":float(balanced_accuracy_score(y,z)),"AUC":float(roc_auc_score(y,p)),"F1":float(f1_score(y,z,zero_division=0)),"FP":int(np.sum((y==0)&z)),"FN":int(np.sum((y==1)&(~z)))}
def main():
    pred=read(PRED);lab=read(LINEAGE);lm={norm(r.get("sha256") or r.get("sample_id")):row_label(r) for r in lab}
    pred_ids=[norm(r.get("sha256")) for r in pred]
    if len(pred)!=1989 or len(lm)!=1989 or len(set(pred_ids))!=1989 or set(pred_ids)!=set(lm):raise RuntimeError("future cohort mismatch")
    rows=[];lines=["CARE-Droid frozen 2017-to-future no-Ego safe authority diagnostic","Future labels were opened only by this summarizer; they did not change the method.",""]
    for scope,idx in [(str(y),[i for i,r in enumerate(pred) if int(r["year"])==y]) for y in (2018,2019,2020,2021)]+[("pooled",list(range(1989)))]:
        y=np.asarray([lm[norm(pred[i]["sha256"])] for i in idx]);p0=np.asarray([float(pred[i]["p_stage1"]) for i in idx]);p=np.asarray([float(pred[i]["p_final"]) for i in idx])
        fixed,broken=fixed_broken(y,p0,p);a=met(y,p0);b=met(y,p);row={"scope":scope,"n":len(idx),"routed":sum(int(pred[i]["route_DI"]) for i in idx),
            "active":sum(int(pred[i]["gate_active"]) for i in idx),**{f"Global24_{k}":v for k,v in a.items()},**{f"CARE_{k}":v for k,v in b.items()},"fixed":fixed,"broken":broken,"net":fixed-broken}
        rows.append(row);lines.append(f"{scope}: Global24_BA={a['BA']:.6f} CARE_BA={b['BA']:.6f} CARE_AUC={b['AUC']:.6f} CARE_F1={b['F1']:.6f} routed={row['routed']} active={row['active']} fixed={fixed} broken={broken} net={fixed-broken}")
    with (OUT/"future_metrics.csv").open("w",encoding="utf-8",newline="") as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines += ["","Diagnostic-only: 2018-2021 were previously inspected and cannot revise threshold, route, representation, or bound."]
    (OUT/"REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8");print("\n".join(lines));print("FROZEN_FUTURE_SAFE_AUTHORITY_DONE")
if __name__=="__main__":main()
