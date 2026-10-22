#!/usr/bin/env python3
import json
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from common import ROOT,BASE,METHOD,read_csv,dump
if not BASE.exists() or BASE.stat().st_size==0:raise RuntimeError("BLOCKED_BY_CONTRACT: missing A0 predictions")
r=read_csv(BASE);required={"sha256","label","fold","Global24__prob","A25__score","A25__routed",METHOD+"__expert_prob",METHOD+"__prob",METHOD+"__bounded_delta"}
missing=required-set(r[0])
if missing:raise RuntimeError("BLOCKED_BY_CONTRACT: missing fields "+str(sorted(missing)))
if len(r)!=799 or sum(int(x["A25__routed"]) for x in r)!=200:raise RuntimeError("BLOCKED_BY_CONTRACT: formal source/route changed")
y=np.asarray([int(x["label"]) for x in r]);b=np.asarray([float(x["Global24__prob"]) for x in r]);p=np.asarray([float(x[METHOD+"__prob"]) for x in r]);bo=(b>=.5)==y;ok=(p>=.5)==y
observed={"BA":float(balanced_accuracy_score(y,p>=.5)),"fixed":int(np.sum(~bo&ok)),"broken":int(np.sum(bo&~ok))}
if abs(observed["BA"]-.9323840852130325)>1e-10 or observed["fixed"]!=11 or observed["broken"]!=7:raise RuntimeError("BLOCKED_BY_CONTRACT: D0 mismatch "+str(observed))
observed["net"]=observed["fixed"]-observed["broken"]
dump(ROOT/"preflight.json",{"source_rows":799,"route":200,"stage1":"P31-RF","router":"RF-anchored MeanOpposition DI25","D0":observed,"fold_sizes":{str(f):sum(int(x["fold"])==f for x in r) for f in range(1,6)},"future_rows_loaded":0})
print("D0_SCORE_CALIBRATION_PREFLIGHT_PASS");print(json.dumps(observed))

