#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score,f1_score,roc_auc_score
from p31_core import fit_calibrated_committee,group_top,mean_opposition,mean_probability,label

OUT=Path("artifacts/care-litelocal-p31-diagnostics-v1")
def read(path):
    with Path(path).open("r",encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def norm(v):return str(v or "").strip().casefold()
def write(path,rows):
    with path.open("w",encoding="utf-8",newline="") as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    cfg=json.loads((OUT/"preflight/resolved_inputs.json").read_text());contract=read(cfg["within_router"]);cm={norm(r["sha256"]):r for r in contract}
    with np.load(OUT/"future_p31_features_2018_2021.npz",allow_pickle=False) as z:ids=np.asarray([norm(x) for x in z["sha256"]]);years=z["year"].astype(int);x=z["role_prevalence"].astype(float)
    y=np.asarray([label(cm[s]["label"]) for s in ids]);folds=np.asarray([int(float(cm[s]["fold"])) for s in ids]);members=np.full((len(ids),4),np.nan);aud=[]
    for year in sorted(set(years)):
        for fold in range(1,6):
            va=np.flatnonzero((years==year)&(folds==fold));tr=np.flatnonzero((years==year)&(folds!=fold));seed=20260925+year*100+fold
            out,a=fit_calibrated_committee(x[tr],y[tr],x[va],seed,inner_folds=4,jobs=4);members[va]=out;aud.extend({"year":year,"outer_fold":fold,**r} for r in a);print(f"P31 within-year {year} fold {fold}/5 done",flush=True)
    if not np.isfinite(members).all():raise RuntimeError("BLOCKED_BY_CONTRACT: incomplete within-year OOF")
    p=mean_probability(members);score=mean_opposition(members,p);route=group_top(score,years,ids,.25);rows=[]
    for i,sid in enumerate(ids):rows.append({"sha256":sid,"year":int(years[i]),"fold":int(folds[i]),"label":int(y[i]),"p_stage1":float(p[i]),"p_lr":float(members[i,0]),"p_rf":float(members[i,1]),"p_xgb":float(members[i,2]),"p_svm":float(members[i,3]),"U_DI":float(score[i]),"route_DI":int(route[i])})
    root=OUT/"within_year";root.mkdir(exist_ok=True);write(root/"oof_predictions.csv",rows);write(root/"fit_audit.csv",aud);metrics=[]
    for scope,idx in [(str(v),years==v) for v in sorted(set(years))]+[("pooled",np.ones(len(y),bool))]:
        pred=p[idx]>=.5;error=(p>=.5)!=y;caught=int(np.sum(error&route&idx));total=int(np.sum(error&idx));metrics.append({"scope":scope,"n":int(idx.sum()),"routed":int(np.sum(route&idx)),"BA":float(balanced_accuracy_score(y[idx],pred)),"AUC":float(roc_auc_score(y[idx],p[idx])),"F1":float(f1_score(y[idx],pred,zero_division=0)),"errors":total,"captured_errors":caught,"error_recall_at_25":caught/max(total,1),"routing_precision":caught/max(int(np.sum(route&idx)),1)})
    write(root/"metrics.csv",metrics);lines=["CARE-Droid P31 Mean-Committee independent same-year five-fold OOF diagnostic","Each year trains and validates only within itself; this is not temporal generalization."]+[f"{r['scope']}: BA={r['BA']:.6f} AUC={r['AUC']:.6f} F1={r['F1']:.6f} errors={r['errors']} routed={r['routed']} captured={r['captured_errors']} recall25={r['error_recall_at_25']:.6f}" for r in metrics];(root/"REPORT.txt").write_text("\n".join(lines)+"\n");print("LITELOCAL_P31_WITHIN_YEAR_DONE");print("\n".join(lines))
if __name__=="__main__":main()
