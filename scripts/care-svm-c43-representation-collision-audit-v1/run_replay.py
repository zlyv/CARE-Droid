#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
import numpy as np
from audit_common import *
from capture_model import CaptureState

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--device",default="cuda"); ap.add_argument("--jobs",type=int,default=4); args=ap.parse_args()
    checkpoint=OUTPUT/"representation_store.npz"
    if checkpoint.is_file() and (OUTPUT/"replay_validation.json").is_file(): print("C43_REPRESENTATION_REPLAY_RESUME",flush=True); return
    sys.path.insert(0,str(FAMILY_SCRIPT)); sys.path.insert(0,str(DIRECT_SCRIPT))
    from common import ENDPOINT_SOURCE,SOURCE_A0_CACHE
    from model_adapter import install_a0,prepare_imports
    from nested_stage1 import NestedStage1
    from model_variants import install_variant
    prepare_imports()
    stage1_script=SCRIPTS/"care-p31-stage1-rf-selection-source-v1"
    if str(stage1_script) not in sys.path:sys.path.insert(0,str(stage1_script))
    from source_label_contract import install_source_label_contract
    import run_care_rolegraph_conditioned_dualstream_v4 as v4
    install_source_label_contract(v4); install_a0(v4,METHOD_REPLAY); install_variant(v4,"U2_DIRECT_STAGE2")
    ids,y,fold,base,route,frozen_expert,frozen_final,_=load_reference(False)
    state=CaptureState(v4,ids,fold,OUTPUT); v4.fit_predict_variant=state.fit_predict
    nested=NestedStage1("SVM",OUTPUT/"audits/replay_nested.json",args.jobs); v4.inner_oof_global24=nested.inner_oof
    out=OUTPUT/"replay"; argv=[sys.argv[0],"--previous-result",str(FAMILY_OUTPUT/"references/SVM"),"--cache",str(ENDPOINT_SOURCE),
        "--region-cache",str(SOURCE_A0_CACHE),"--variants","RGCondAttn","--seeds","0,1,2","--max-delta","2.25",
        "--device",args.device,"--output",str(out),"--keep-weight","1","--fix-weight","1","--keep-consistency","0"]
    sys.argv=argv; v4.main()
    rows=read_csv(out/"source_predictions.csv"); replay_order=np.asarray([norm(r["sha256"]) for r in rows]); by={norm(r["sha256"]):r for r in rows}; field=METHOD_REPLAY+"__expert_prob"
    replay=np.asarray([float(by[s][field]) for s in ids]); final=base.copy(); final[route]=replay[route]
    m=metrics(y,base,final,route); max_abs=float(np.max(np.abs(replay-frozen_expert))); mean_abs=float(np.mean(np.abs(replay-frozen_expert)))
    checks={"fold_calls==5":state.next_fold==6,"nested_calls==5":len(nested.calls)==5,"manifest_order_exact":bool(np.array_equal(replay_order,ids)),"direct_metrics_reproduced":(m["fixed"],m["broken"],m["net"])==(49,17,32),
        "routed_decisions_exact":bool(np.array_equal(replay[route]>=.5,frozen_expert[route]>=.5)),"max_probability_abs_diff<=1e-4":max_abs<=1e-4}
    if not all(checks.values()):raise RuntimeError("BLOCKED_BY_CONTRACT: Direct-C43 replay mismatch "+repr({"checks":checks,"metrics":m,"max_abs":max_abs}))
    state.save(checkpoint); dump_json(OUTPUT/"replay_validation.json",{"status":"PASS","checks":checks,"metrics":m,"max_abs_probability_difference":max_abs,"mean_abs_probability_difference":mean_abs,"future_rows_loaded":0})
    print("C43_REPRESENTATION_REPLAY_PASS max_abs=%.8g"%max_abs,flush=True)
if __name__=="__main__":main()
