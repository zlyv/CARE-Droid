#!/usr/bin/env python3
from __future__ import annotations
import json
import numpy as np
from common import ROOT,SOURCE,METHOD,BASE_FIELD,ROUTE_FIELD,ROUTER_SCORE_FIELD,MAX_DELTA,read_csv,write_csv,dump,metrics,norm
rows=read_csv(SOURCE);required={"sha256","fold","label",BASE_FIELD,ROUTE_FIELD,ROUTER_SCORE_FIELD,METHOD+"__expert_prob",METHOD+"__prob",METHOD+"__bounded_delta"};missing=required-set(rows[0])
if missing:raise RuntimeError("FORMAL_REPRODUCTION_FAILED: missing fields "+str(sorted(missing)))
sha=np.asarray([norm(r["sha256"]) for r in rows]);fold=np.asarray([int(r["fold"]) for r in rows]);y=np.asarray([int(r["label"]) for r in rows]);base=np.asarray([float(r[BASE_FIELD]) for r in rows]);route=np.asarray([int(r[ROUTE_FIELD]) for r in rows],bool);score=np.asarray([float(r[ROUTER_SCORE_FIELD]) for r in rows]);candidate=np.asarray([float(r[METHOD+"__expert_prob"]) for r in rows]);saved=np.asarray([float(r[METHOD+"__prob"]) for r in rows]);delta=np.asarray([float(r[METHOD+"__bounded_delta"]) for r in rows])
errors=[]
if len(rows)!=799:errors.append(f"rows={len(rows)}")
if (int(np.sum(y==0)),int(np.sum(y==1)))!=(400,399):errors.append("class counts changed")
if [int(np.sum(fold==f)) for f in range(1,6)]!=[160,160,160,160,159]:errors.append("fold sizes changed")
if [int(np.sum(route&(fold==f))) for f in range(1,6)]!=[40,40,40,40,40]:errors.append("formal route counts changed")
if not all(np.isfinite(x).all() for x in (base,score,candidate,saved,delta)):errors.append("NaN/Inf")
if np.max(np.abs(delta))>MAX_DELTA+1e-8:errors.append("bounded delta exceeds max_delta")
nonrouted=~route;nonrouted_candidate_count=int(np.sum(np.abs(candidate[nonrouted]-base[nonrouted])>1e-12))
if nonrouted_candidate_count<int(.5*np.sum(nonrouted)):errors.append("C43 candidate was not materially saved for enough non-routed rows; A1/A3 cannot be derived")
reconstructed=np.where(route,candidate,base);max_abs_diff=float(np.max(np.abs(reconstructed-saved)))
if max_abs_diff>1e-8:errors.append(f"saved final is not route-composed candidate max_diff={max_abs_diff}")
result=metrics(y,base,saved,route);expected={"BA":.9323840852130325,"AUC":.9756109022556392,"F1":.9305912596401028,"fixed":11,"broken":7,"net":4}
for key,value in expected.items():
    if abs(float(result[key])-value)>1e-10:errors.append(f"{key}: observed={result[key]} expected={value}")
out=[]
for i,r in enumerate(rows):out.append({"sha256":sha[i],"fold":fold[i],"label":y[i],"stage1_prob":base[i],"stage1_pred":int(base[i]>=.5),"router_score":score[i],"routed":int(route[i]),"candidate_prob":candidate[i],"bounded_prob":candidate[i],"bounded_delta":delta[i],"final_prob":saved[i],"final_pred":int(saved[i]>=.5)})
write_csv(ROOT/"formal_predictions.csv",out)
audit={"status":"FAIL" if errors else "PASS","source_rows":len(rows),"class_counts":{"benign":int(np.sum(y==0)),"malware":int(np.sum(y==1))},"fold_sizes":[int(np.sum(fold==f)) for f in range(1,6)],"route_per_fold":[int(np.sum(route&(fold==f))) for f in range(1,6)],"nonrouted_rows_with_material_candidate":nonrouted_candidate_count,"prediction_reconstruction_max_abs_diff":max_abs_diff,"observed":result,"expected":expected,"errors":errors,"future_rows_loaded":0}
dump(ROOT/"formal_reproduction_audit.json",audit)
if errors:
    print("FORMAL_REPRODUCTION: FAIL");print(json.dumps(audit));raise RuntimeError("FORMAL_REPRODUCTION_FAILED")
print("FORMAL_REPRODUCTION: PASS");print(json.dumps(result))

