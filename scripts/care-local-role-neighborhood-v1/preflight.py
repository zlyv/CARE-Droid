#!/usr/bin/env python3
import json
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from common import ROOT,BASE,CACHE,METHOD,ROLE_NAMES,read_csv,dump,norm
missing=[str(p) for p in (BASE,CACHE) if not p.exists() or p.stat().st_size==0]
if missing:raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty inputs:\n- "+"\n- ".join(missing))
rows=read_csv(BASE);required={"sha256","fold","label","Global24__prob","A25__routed","A25__score",METHOD+"__prob",METHOD+"__bounded_delta"};absent=required-set(rows[0])
if absent:raise RuntimeError("BLOCKED_BY_CONTRACT: missing columns "+str(sorted(absent)))
if len(rows)!=799 or sum(int(r["A25__routed"]) for r in rows)!=200:raise RuntimeError("BLOCKED_BY_CONTRACT: source/route changed")
y=np.asarray([int(r["label"]) for r in rows]);base=np.asarray([float(r["Global24__prob"]) for r in rows]);d0=np.asarray([float(r[METHOD+"__prob"]) for r in rows]);bo=(base>=.5)==y;ok=(d0>=.5)==y
obs={"BA":float(balanced_accuracy_score(y,d0>=.5)),"fixed":int(np.sum(~bo&ok)),"broken":int(np.sum(bo&~ok))};obs["net"]=obs["fixed"]-obs["broken"]
if abs(obs["BA"]-.9323840852130325)>1e-10 or (obs["fixed"],obs["broken"])!=(11,7):raise RuntimeError("BLOCKED_BY_CONTRACT: L0 mismatch "+str(obs))
with np.load(CACHE,allow_pickle=False) as z:
    keys=set(z.files)
    if not {"sha256","host_values","host_offsets"}<=keys:raise RuntimeError("BLOCKED_BY_CONTRACT: cache schema changed")
    if z["host_values"].shape[1]<74 or len(z["host_offsets"])!=800:raise RuntimeError("BLOCKED_BY_CONTRACT: invalid Host cache shape")
    if not np.array_equal(np.asarray([norm(r["sha256"]) for r in rows]),np.asarray([norm(x) for x in z["sha256"][:799]])):raise RuntimeError("BLOCKED_BY_CONTRACT: SHA alignment failed")
audit={"source_rows":799,"fold_sizes":{str(f):sum(int(r["fold"])==f for r in rows) for f in range(1,6)},"route":200,"L0":obs,"representation_source":str(CACHE),"representation_definition":"Compact43 Host content grouped by SelfRole31","representation_label_free":True,"learned_hidden_embedding_available":False,"role_count":len(ROLE_NAMES),"outer_test_in_memory":False,"test_test_retrieval":False,"future_rows_loaded":0}
dump(ROOT/"preflight.json",audit)
print("LOCAL_ROLE_NEIGHBOR_PREFLIGHT_PASS");print("LOCAL_ROLE_REPRESENTATION_AUDIT: fold-safe label-free Compact43 Host content grouped by SelfRole31; learned hidden C43 embeddings unavailable");print(json.dumps(obs))

