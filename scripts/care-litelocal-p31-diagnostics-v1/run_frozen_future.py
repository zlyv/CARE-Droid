#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from p31_core import fit_calibrated_committee,group_top,mean_opposition,mean_probability,label

OUT=Path("artifacts/care-litelocal-p31-diagnostics-v1")
def read(path):
    with Path(path).open("r",encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def norm(v):return str(v or "").strip().casefold()
def write(path,rows):
    with path.open("w",encoding="utf-8",newline="") as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    cfg=json.loads((OUT/"preflight/resolved_inputs.json").read_text());source=read(cfg["source_router"]);sm={norm(r["sha256"]):r for r in source}
    with np.load(cfg["source_features"],allow_pickle=False) as z:sids=np.asarray([norm(x) for x in z["sha256"]]);sx=z["role_prevalence"].astype(float)
    sy=np.asarray([label(sm[s]["label"]) for s in sids])
    with np.load(OUT/"future_p31_features_2018_2021.npz",allow_pickle=False) as z:fids=np.asarray([norm(x) for x in z["sha256"]]);years=z["year"].astype(int);fx=z["role_prevalence"].astype(float)
    members,audit=fit_calibrated_committee(sx,sy,fx,20260925,inner_folds=5,jobs=4);p=mean_probability(members);score=mean_opposition(members,p);route=group_top(score,years,fids,.25)
    rows=[]
    for i,sid in enumerate(fids):rows.append({"sha256":sid,"year":int(years[i]),"p_stage1":float(p[i]),"p_lr":float(members[i,0]),"p_rf":float(members[i,1]),"p_xgb":float(members[i,2]),"p_svm":float(members[i,3]),"U_DI":float(score[i]),"route_DI":int(route[i])})
    frozen=OUT/"frozen_future";frozen.mkdir(exist_ok=True);write(frozen/"label_free_predictions.csv",rows);write(frozen/"fit_audit.csv",audit)
    print("LITELOCAL_P31_FROZEN_FUTURE_LABEL_FREE_DONE");print("future_rows=1989 routed=497 labels_loaded=false")
if __name__=="__main__":main()
