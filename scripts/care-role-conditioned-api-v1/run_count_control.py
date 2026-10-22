#!/usr/bin/env python3
import csv,json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,f1_score
from role_api_core import write_csv,norm
ROOT=Path("artifacts/care-role-conditioned-api-v1")
def read(p):
    with Path(p).open(encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def main():
    ref=read("artifacts/care-p31-stage1-rf-selection-source-v1/preflight/C_RF_RFROUTE/source_predictions.csv");audit={norm(r["sha256"]):r for r in read(ROOT/"cache/apk_api_audit.csv")};ids=[norm(r["sha256"]) for r in ref];y=np.asarray([int(float(r["label"])) for r in ref]);fold=np.asarray([int(float(r["fold"])) for r in ref]);fields=["real_host_count","sensitive_api_occurrence_count","unique_api_count","role_count"]+[f"role_api_count_{i}" for i in range(31)];x=np.asarray([[float(audit[s][f]) for f in fields] for s in ids]);x=np.log1p(x);p=np.full(799,np.nan)
    for f in range(1,6):
        tr=fold!=f;va=fold==f;m=RandomForestClassifier(n_estimators=400,min_samples_leaf=3,class_weight="balanced_subsample",max_features="sqrt",random_state=31000+f,n_jobs=4);m.fit(x[tr],y[tr]);p[va]=m.predict_proba(x[va])[:,1]
    row={"method":"COUNT_ONLY","BA":balanced_accuracy_score(y,p>=.5),"AUC":roc_auc_score(y,p),"F1":f1_score(y,p>=.5)};write_csv(ROOT/"count_only_results.csv",[row]);write_csv(ROOT/"count_only_predictions.csv",[{"sha256":s,"fold":int(fold[i]),"label":int(y[i]),"prob":float(p[i])} for i,s in enumerate(ids)]);print("COUNT_CONTROL_DONE",json.dumps(row))
if __name__=="__main__":main()
