#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from p31_core import label

OUT=Path("artifacts/care-litelocal-p31-diagnostics-v1")
def read(path):
    with Path(path).open("r",encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def norm(v):return str(v or "").strip().casefold()
def model(seed):return Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=1,solver="liblinear",max_iter=5000,random_state=seed))])
def bootstrap(y,a,b,n=2000):
    rng=np.random.default_rng(20260928);idx=[np.flatnonzero(y==v) for v in (0,1)];out=[]
    for _ in range(n):
        s=np.concatenate([rng.choice(x,len(x),replace=True) for x in idx]);out.append(roc_auc_score(y[s],a[s])-roc_auc_score(y[s],b[s]))
    return float(np.mean(out)),float(np.quantile(out,.025)),float(np.quantile(out,.975))
def main():
    cfg=json.loads((OUT/"preflight/resolved_inputs.json").read_text());router=read(cfg["source_router"]);rm={norm(r["sha256"]):r for r in router};locked=read(cfg["source_predictions"]);pm={norm(r["sha256"]):r for r in locked}
    with np.load(cfg["source_features"],allow_pickle=False) as z:ids=np.asarray([norm(x) for x in z["sha256"]]);x=z["role_prevalence"].astype(float);audit=z["audit_values"].astype(float);names=[str(v) for v in z["audit_names"]]
    y=np.asarray([label(rm[s]["label"]) for s in ids]);folds=np.asarray([int(float(rm[s]["fold"])) for s in ids]);nuisance=np.log1p(np.maximum(audit[:,[names.index(v) for v in ("apk_bytes","class_count","method_ref_count","sensitive_method_count","mapped_invoke_count")]],0));size=np.full(len(y),np.nan);combined=np.full(len(y),np.nan)
    for fold in range(1,6):
        tr=np.flatnonzero(folds!=fold);va=np.flatnonzero(folds==fold);a=model(20260928+fold);a.fit(nuisance[tr],y[tr]);size[va]=a.predict_proba(nuisance[va])[:,1];b=model(20261028+fold);b.fit(np.hstack([x[tr],nuisance[tr]]),y[tr]);combined[va]=b.predict_proba(np.hstack([x[va],nuisance[va]]))[:,1]
    locked_p=np.asarray([float(pm[s]["p_P31__MEAN_COMMITTEE"]) for s in ids]);mean,lo,hi=bootstrap(y,combined,size);report={"P31_mean_committee_BA":float(balanced_accuracy_score(y,locked_p>=.5)),"P31_mean_committee_AUC":float(roc_auc_score(y,locked_p)),"size_only_BA":float(balanced_accuracy_score(y,size>=.5)),"size_only_AUC":float(roc_auc_score(y,size)),"P31_plus_size_BA":float(balanced_accuracy_score(y,combined>=.5)),"P31_plus_size_AUC":float(roc_auc_score(y,combined)),"incremental_AUC":float(roc_auc_score(y,combined)-roc_auc_score(y,size)),"bootstrap_mean_delta_AUC":mean,"bootstrap_CI_low":lo,"bootstrap_CI_high":hi,"semantic_increment_supported":bool(lo>0)}
    root=OUT/"source_shortcut";root.mkdir(exist_ok=True);(root/"audit.json").write_text(json.dumps(report,indent=2)+"\n");lines=["CARE-Droid P31-specific size-conditioned source audit"]+[f"{k}={v}" for k,v in report.items()];(root/"REPORT.txt").write_text("\n".join(lines)+"\n");print("LITELOCAL_P31_SOURCE_SHORTCUT_DONE");print("\n".join(lines))
if __name__=="__main__":main()
