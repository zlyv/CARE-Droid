#!/usr/bin/env python3
"""Immediate primary M0/M1 source report before the auxiliary MalBen run."""
import csv
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score, confusion_matrix
from experiment_contract import read_csv, write_csv

root=Path("artifacts/care-di25-hostego-v1")
rows=[]; saved={}
for arm,folder in (("M0","M0_host_anomaly_bounded"),("M1","M1_hostego_anomaly_bounded")):
    data=read_csv(root/"source"/folder/"source_predictions.csv")
    prob=[c for c in data[0] if c.startswith("DI25__") and c.endswith("__prob")][0]
    direct_col=prob[:-6]+"__direct_stage2_probability"
    y=np.asarray([int(r["label"]) for r in data]); base=np.asarray([float(r["Global24__prob"]) for r in data])
    route=np.asarray([int(r["A25__routed"])==1 for r in data]); bounded=np.asarray([float(r[prob]) for r in data])
    direct=np.where(route,np.asarray([float(r[direct_col]) for r in data]),base)
    saved[arm]=(y,bounded,direct)
    for head,p in (("Bounded",bounded),("Direct",direct)):
        pred=p>=.5; bp=base>=.5; tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
        fixed=int(np.sum((bp!=y)&(pred==y))); broken=int(np.sum((bp==y)&(pred!=y)))
        rows.append({"arm":arm,"head":head,"BA":balanced_accuracy_score(y,pred),"AUC":roc_auc_score(y,p),"F1":f1_score(y,pred),
                     "TN":tn,"FP":fp,"FN":fn,"TP":tp,"fixed":fixed,"broken":broken,"net":fixed-broken})
write_csv(root/"results/M0_M1_primary_comparison.csv",rows)
lines=["CARE-Droid primary M0/M1 source comparison",""]+[f"{r['arm']} {r['head']}: BA={r['BA']:.6f} AUC={r['AUC']:.6f} F1={r['F1']:.6f} fixed={r['fixed']} broken={r['broken']} net={r['net']}" for r in rows]
(root/"results/M0_M1_REPORT.txt").parent.mkdir(parents=True,exist_ok=True)
(root/"results/M0_M1_REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
print("\n".join(lines))
