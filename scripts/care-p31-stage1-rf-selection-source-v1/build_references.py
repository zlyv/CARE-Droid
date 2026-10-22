#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,f1_score
from stage1_core import mean_opposition,foldwise_top25

ROOT=Path("artifacts/care-p31-stage1-rf-selection-source-v1")
SRC=Path("artifacts/care-litelocal-stage1-attribution-source-v1/source_member_oof_predictions.csv")
MEMBERS=("LR","RF","XGB","SVM")
PROTOCOLS={
 "A_LR_LRROUTE":("LR","LR"),
 "B_RF_LRROUTE":("RF","LR"),
 "C_RF_RFROUTE":("RF","RF"),
 "D_MEAN_MEANROUTE":("MEAN","MEAN"),
}

def read(path):
    with path.open(encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))

def main():
    rows=read(SRC)
    if len(rows)!=799: raise RuntimeError("BLOCKED_BY_CONTRACT: member table changed")
    sha=np.asarray([r["sha256"].strip().casefold() for r in rows]); y=np.asarray([int(float(r["label"])) for r in rows]); fold=np.asarray([int(float(r["fold"])) for r in rows])
    members=np.asarray([[float(r[f"p_P31__{m}"]) for m in MEMBERS] for r in rows])
    if members.shape!=(799,4) or not np.isfinite(members).all() or len(set(sha.tolist()))!=799:
        raise RuntimeError("BLOCKED_BY_CONTRACT: invalid locked probabilities")
    probs={"LR":members[:,0],"RF":members[:,1],"MEAN":members.mean(axis=1)}
    summaries=[]
    for arm,(base_mode,anchor_mode) in PROTOCOLS.items():
        base=probs[base_mode]; anchor=probs[anchor_mode]
        score=mean_opposition(members,anchor); route=foldwise_top25(score,fold,sha)
        pred=base>=.5; err=pred!=y; captured=err&route
        target=ROOT/"preflight"/arm/"source_predictions.csv"; target.parent.mkdir(parents=True,exist_ok=True)
        out=[]
        for i in range(799):
            out.append({"sha256":sha[i],"label":int(y[i]),"fold":int(fold[i]),
                "Global24__prob":float(base[i]),"Global24__A25__JointS7_ordinary__prob":float(base[i]),
                "A25__score":float(score[i]),"A25__routed":int(route[i]),
                "stage1_mode":base_mode,"router_anchor":anchor_mode})
        with target.open("w",encoding="utf-8",newline="") as h:
            w=csv.DictWriter(h,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
        summaries.append({"arm":arm,"stage1":base_mode,"router_anchor":anchor_mode,
            "BA":float(balanced_accuracy_score(y,pred)),"AUC":float(roc_auc_score(y,base)),
            "F1":float(f1_score(y,pred)),"errors":int(err.sum()),"routed":int(route.sum()),
            "captured_errors":int(captured.sum()),"error_recall25":float(captured.sum()/max(err.sum(),1)),
            "precision25":float(captured.sum()/route.sum()),"FN":int(np.sum((y==1)&~pred)),
            "FP":int(np.sum((y==0)&pred))})
    (ROOT/"preflight"/"protocol.json").write_text(json.dumps({"source_only":True,
        "future_rows_loaded":0,"P31_dimensions":31,"protocols":PROTOCOLS,
        "route_budget":.25,"route_tiebreak":"sha256"},indent=2)+"\n",encoding="utf-8")
    with (ROOT/"stage1_router_metrics.csv").open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    print("P31_RF_SELECTION_REFERENCES_PASS")
    for r in summaries: print(f"{r['arm']}: BA={r['BA']:.6f} errors={r['errors']} captured={r['captured_errors']}/{r['errors']} recall25={r['error_recall25']:.6f}")

if __name__=="__main__":main()
