#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from common import ROOT,BASE,CACHE,METHOD,read_csv,dump,norm
req=[BASE,CACHE,Path("artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz")]
miss=[str(p) for p in req if not p.exists() or p.stat().st_size==0]
if miss:raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty inputs:\n- "+"\n- ".join(miss))
nested=[Path("artifacts/care-role-reliability-gate-v1/nested")/f"outer_{f}.npz" for f in range(1,6)]
fallback=Path("scripts/care-p31-stage1-rf-selection-source-v1/stage1_core.py")
if not all(p.exists() and p.stat().st_size>0 for p in nested) and not fallback.exists():raise RuntimeError("BLOCKED_BY_CONTRACT: neither all strict nested gate contexts nor Stage-I rebuild code is available")
r=read_csv(BASE)
if len(r)!=799 or sum(int(x["A25__routed"]) for x in r)!=200:raise RuntimeError("BLOCKED_BY_CONTRACT: formal source/route changed")
if METHOD+"__prob" not in r[0]:raise RuntimeError("BLOCKED_BY_CONTRACT: A0 baseline missing")
y=np.asarray([int(x["label"]) for x in r]);base=np.asarray([float(x["Global24__prob"]) for x in r]);d0=np.asarray([float(x[METHOD+"__prob"]) for x in r]);bo=(base>=.5)==y;ok=(d0>=.5)==y
observed=(balanced_accuracy_score(y,d0>=.5),int(np.sum(~bo&ok)),int(np.sum(bo&~ok)))
if abs(observed[0]-.9323840852130325)>1e-10 or observed[1:]!=(11,7):raise RuntimeError(f"BLOCKED_BY_CONTRACT: D0 baseline mismatch {observed}")
with np.load(CACHE,allow_pickle=False) as z:
 if z["host_values"].shape[1]<105 or len(z["sha256"])<799:raise RuntimeError("BLOCKED_BY_CONTRACT: A0 Host cache changed")
 if not np.array_equal(np.asarray([norm(x["sha256"]) for x in r]),np.asarray([norm(x) for x in z["sha256"][:799]])):raise RuntimeError("BLOCKED_BY_CONTRACT: SHA mismatch")
a={"source_rows":799,"fold_sizes":{str(f):sum(int(x["fold"])==f for x in r) for f in range(1,6)},"stage1":"P31 calibrated RF","router":"RF-anchored MeanOpposition DI25 Top25%","stage2_baseline":"A0 Role C43-RQ bounded","D0":{"BA":observed[0],"fixed":observed[1],"broken":observed[2],"net":observed[1]-observed[2]},"prototype_input":"Compact43 Host content grouped by SelfRole31","minimum_per_class_role_support":5,"metric":"cosine similarity difference: malware minus benign","max_delta":2.25,"future_rows_loaded":0}
dump(ROOT/"preflight.json",a);print("CLASS_CONDITIONAL_ROLE_PREFLIGHT_PASS");print(json.dumps(a))

