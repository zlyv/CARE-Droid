#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
from common import ROOT,BASE,CACHE,METHOD,read_csv,dump,norm
req=[BASE,CACHE,Path("artifacts/care-p31-stage1-rf-selection-source-v1/preflight/C_RF_RFROUTE/source_predictions.csv"),Path("artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz")]
missing=[str(p) for p in req if not p.exists() or p.stat().st_size==0]
if missing:raise RuntimeError("BLOCKED_BY_CONTRACT: missing inputs:\n- "+"\n- ".join(missing))
rows=read_csv(BASE)
if len(rows)!=799 or len({norm(r["sha256"]) for r in rows})!=799:raise RuntimeError("BLOCKED_BY_CONTRACT: source cohort changed")
needed={"sha256","label","fold","Global24__prob","A25__score","A25__routed",METHOD+"__prob",METHOD+"__bounded_delta"}
if not needed<=set(rows[0]):raise RuntimeError("BLOCKED_BY_CONTRACT: A0 prediction schema mismatch "+repr(sorted(needed-set(rows[0]))))
if sum(int(r["A25__routed"]) for r in rows)!=200:raise RuntimeError("BLOCKED_BY_CONTRACT: route is not DI25=200")
with np.load(CACHE,allow_pickle=False) as z:
    keys=set(z.files);shape=list(z["host_values"].shape);sha=z["sha256"].astype(str)
    if not {"sha256","host_values","host_offsets"}<=keys or len(sha)<799:raise RuntimeError("BLOCKED_BY_CONTRACT: Role Host cache schema mismatch")
audit={"source_rows":799,"fold_sizes":{str(f):sum(int(r["fold"])==f for r in rows) for f in range(1,6)},"stage1":"P31 calibrated RF","router":"RF-anchored MeanOpposition DI25 Top25%","stage2":"A0 Role-only C43-RQ bounded correction","route":200,"host_values_shape":shape,"strict_nested_gate_required":True,"future_rows_loaded":0}
dump(ROOT/"preflight.json",audit);print("ROLE_RELIABILITY_GATE_PREFLIGHT_PASS");print(json.dumps(audit))

