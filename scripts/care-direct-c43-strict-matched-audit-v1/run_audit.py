#!/usr/bin/env python3
from __future__ import annotations

import argparse, sys
import numpy as np
from audit_common import *
from capture_interventions import InterventionState

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--device",default="cuda"); ap.add_argument("--jobs",type=int,default=4); ap.add_argument("--repeats",type=int,default=12); cli=ap.parse_args()
    checkpoint=OUTPUT/"intervention_probabilities.npz"
    if checkpoint.is_file() and (OUTPUT/"replay_validation.json").is_file(): print("DIRECT_C43_STRICT_MATCHED_RESUME",flush=True); return
    sys.path.insert(0,str(FAMILY_SCRIPT)); sys.path.insert(0,str(DIRECT_SCRIPT))
    from common import ENDPOINT_SOURCE,SOURCE_A0_CACHE
    from model_adapter import install_a0,prepare_imports
    from nested_stage1 import NestedStage1
    from model_variants import install_variant
    prepare_imports(); stage1=SCRIPTS/"care-p31-stage1-rf-selection-source-v1"
    if str(stage1) not in sys.path: sys.path.insert(0,str(stage1))
    from source_label_contract import install_source_label_contract
    import run_care_rolegraph_conditioned_dualstream_v4 as v4
    install_source_label_contract(v4); install_a0(v4,METHOD_REPLAY); install_variant(v4,"U2_DIRECT_STAGE2")
    ids,y,fold,base,route,frozen_expert,_,_=load_reference(False)
    with np.load(P31,allow_pickle=False) as archive:
        p31_ids=[norm(x) for x in archive["sha256"]]
        p31_values=np.asarray(archive["role_prevalence"],float)
    p31_position={sid:i for i,sid in enumerate(p31_ids)}
    if p31_values.shape[1] != 31 or any(sid not in p31_position for sid in ids):
        raise RuntimeError("BLOCKED_BY_CONTRACT: frozen P31 alignment failed")
    aligned_p31=p31_values[[p31_position[sid] for sid in ids]]
    state=InterventionState(v4,ids,fold,aligned_p31,OUTPUT,cli.repeats); v4.fit_predict_variant=state.fit_predict
    nested=NestedStage1("SVM",OUTPUT/"audits/replay_nested.json",cli.jobs); v4.inner_oof_global24=nested.inner_oof
    out=OUTPUT/"replay"; argv=[sys.argv[0],"--previous-result",str(FAMILY_OUTPUT/"references/SVM"),"--cache",str(ENDPOINT_SOURCE),
      "--region-cache",str(SOURCE_A0_CACHE),"--variants","RGCondAttn","--seeds","0,1,2","--max-delta","2.25",
      "--device",cli.device,"--output",str(out),"--keep-weight","1","--fix-weight","1","--keep-consistency","0"]
    sys.argv=argv; v4.main(); state.save(checkpoint); write_csv(OUTPUT/"matched_donors.csv",state.donor_rows)
    saved=np.load(checkpoint,allow_pickle=False); replay=saved["FULL__mean"]
    final=base.copy(); final[route]=replay[route]; m=metrics(y,base,final,route)
    max_abs=float(np.max(np.abs(replay-frozen_expert)))
    row_abs=float(np.max(np.abs(saved["N0_ROW_ORDER__mean"]-replay)))
    checks={"fold_calls==5":state.next_fold==6,"nested_calls==5":len(nested.calls)==5,
      "direct_metrics_reproduced":(m["fixed"],m["broken"],m["net"])==(49,17,32),
      "max_probability_abs_diff<=1e-4":max_abs<=1e-4,"row_order_max_abs_diff<=1e-5":row_abs<=1e-5}
    if not all(checks.values()): raise RuntimeError("BLOCKED_BY_CONTRACT: replay mismatch "+repr(checks))
    dump_json(OUTPUT/"replay_validation.json",{"status":"PASS","checks":checks,"metrics":m,
      "max_abs_probability_difference":max_abs,"row_order_max_abs_difference":row_abs,
      "strict_match_eligible":int(saved["STRICT_MATCH__eligible"].sum()),"future_rows_loaded":0})
    print("DIRECT_C43_STRICT_MATCHED_REPLAY_PASS",flush=True)
if __name__=="__main__": main()

