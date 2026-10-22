#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,f1_score
from lr_core import mean_opposition,foldwise_top25

ROOT=Path("artifacts/care-litelocal-p31lr-c43-prov6-source-v1")
SRC=Path("artifacts/care-litelocal-stage1-attribution-source-v1/source_member_oof_predictions.csv")
MEMBERS=("LR","RF","XGB","SVM")
def read(p):
    with p.open("r",encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def main():
    rows=read(SRC)
    if len(rows)!=799:raise RuntimeError("BLOCKED_BY_CONTRACT: locked member table changed")
    sha=np.asarray([r["sha256"].strip().casefold() for r in rows]); year=np.asarray([int(float(r["year"])) for r in rows]); fold=np.asarray([int(float(r["fold"])) for r in rows]); y=np.asarray([int(float(r["label"])) for r in rows])
    members=np.asarray([[float(r[f"p_P31__{m}"]) for m in MEMBERS] for r in rows]); p=members[:,0]
    if len(set(sha))!=799 or set(year.tolist())!={2017}:raise RuntimeError("BLOCKED_BY_CONTRACT: not strict 2017")
    if Counter(fold)!=Counter({1:160,2:160,3:160,4:160,5:159}):raise RuntimeError("BLOCKED_BY_CONTRACT: folds changed")
    score=mean_opposition(members,p); route=foldwise_top25(score,fold,sha); pred=p>=.5; err=pred!=y
    ba=float(balanced_accuracy_score(y,pred)); auc=float(roc_auc_score(y,p)); f1=float(f1_score(y,pred)); captured=int(np.sum(err&route))
    if not (abs(ba-.8584774436090226)<1e-12 and abs(auc-.9331203007518798)<1e-12 and int(err.sum())==113 and captured==99):raise RuntimeError("BLOCKED_BY_CONTRACT: P31-LR locked result mismatch")
    out=[]
    for i in range(799):
        out.append({"sha256":sha[i],"label":int(y[i]),"fold":int(fold[i]),"Global24__prob":float(p[i]),"Global24__A25__JointS7_ordinary__prob":float(p[i]),"A25__score":float(score[i]),"A25__routed":int(route[i]),"P31_LR__prob":float(p[i]),"P31_LR_COMMITTEE_OPPOSITION25__score":float(score[i]),"P31_LR_COMMITTEE_OPPOSITION25__routed":int(route[i])})
    target=ROOT/"preflight/lr_reference/source_predictions.csv";target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("w",encoding="utf-8",newline="") as h:w=csv.DictWriter(h,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    contract={"stage1":"calibrated P31 Logistic Regression only","committee_role":"routing only","router":"LR-anchored mean opposition, foldwise top 25%","source_rows":799,"year":2017,"route":200,"errors":113,"captured_errors":99,"BA":ba,"AUC":auc,"F1":f1}
    (ROOT/"preflight/lr_contract.json").write_text(json.dumps(contract,indent=2)+"\n",encoding="utf-8")
    print("P31LR_PROV6_REFERENCE_PASS");print(f"source_rows=799 year=2017 route=200 captured_errors={captured}/113");print(f"P31_LR BA={ba:.12f} AUC={auc:.12f} F1={f1:.12f}")
if __name__=="__main__":main()
