#!/usr/bin/env python3
import hashlib,sys
import numpy as np
from ablation_common import *

STRICT_DECISION=PROJECT/"outputs/care-direct-c43-strict-matched-audit-v1/final_decision.json"

def main():
    if not OUTPUT.is_relative_to((PROJECT/"outputs").resolve()): raise RuntimeError("BLOCKED_BY_CONTRACT: unsafe output")
    required=[FORMAL,PRIOR_OUTPUT/"references/SVM/source_predictions.csv",PRIOR_SCRIPT/"run_source_arm.py",DIRECT_REFERENCE,STRICT_DECISION]
    missing=[str(p) for p in required if not p.is_file()]
    if missing: raise RuntimeError("BLOCKED_BY_CONTRACT: missing prerequisites "+repr(missing))
    strict=read_json(STRICT_DECISION)
    if strict.get("status")!="STRICT_MATCHED_ROLEGRAPH_CONTRIBUTION_CONFIRMED":
        raise RuntimeError("BLOCKED_BY_CONTRACT: strict matched RoleGraph gate did not pass")
    OUTPUT.mkdir(parents=True,exist_ok=True)
    rows=formal_rows(); y=np.asarray([int(float(r["label"])) for r in rows]); fold=np.asarray([int(float(r["fold"])) for r in rows])
    base=np.asarray([float(r["stage1_prob"]) for r in rows]); route=np.asarray([int(float(r["routed"])) for r in rows],bool)
    stage1=metrics(y,base,base,route)
    checks={"rows==799":len(rows)==799,"folds_fixed":[int(np.sum(fold==f)) for f in range(1,6)]==[160,160,160,160,159],
      "route==200":int(route.sum())==200,"stage1_errors==97":stage1["errors"]==97,"strict_matched_gate":True}
    if not all(checks.values()): raise RuntimeError("BLOCKED_BY_CONTRACT: frozen source changed "+repr(checks))
    assets=[asset(p,purpose) for p,purpose in [(FORMAL,"frozen SVM source predictions"),
      (PRIOR_OUTPUT/"references/SVM/source_predictions.csv","frozen SVM Stage-I/router reference"),
      (DIRECT_REFERENCE,"frozen Direct-C43 reference"),(STRICT_DECISION,"strict matched authorization")]]
    dump_json(OUTPUT/"baseline_asset_manifest_before.json",assets)
    protocol={"name":"Direct-C43 RoleGraph query/readout factorial v1","domain":"2017 fixed outer-fold OOF only",
      "arms":{arm:{"role_query":int(arm[1]),"role_readout":int(arm[2])} for arm in ARMS},
      "stage1":"calibrated P31-SVM","router":"SVM MeanOpposition Top25%","stage2":"Direct-C43",
      "seeds":[0,1,2],"threshold":0.5,"same_parameter_set":True,
      "switches":"fixed non-trainable buffers; each arm retrained with identical architecture and protocol",
      "selection":"source attribution only; no threshold tuning and no future selection","future_rows_loaded":0}
    dump_json(OUTPUT/"protocol.json",protocol)
    (OUTPUT/"protocol_hash.txt").write_text(hashlib.sha256((OUTPUT/"protocol.json").read_bytes()).hexdigest()+"\n",encoding="utf-8")
    dump_json(OUTPUT/"preflight.json",{"status":"PASS","checks":checks,"stage1":stage1,"strict_gate":strict,"future_rows_loaded":0})
    print("DIRECT_C43_ROLEPATH_FACTORIAL_PREFLIGHT_PASS",flush=True)

if __name__=="__main__":main()
