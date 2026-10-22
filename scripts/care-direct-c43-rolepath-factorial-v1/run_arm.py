#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from ablation_common import ARMS, METHOD, OUTPUT, PRIOR_OUTPUT, PRIOR_SCRIPT
from direct_variant import install_direct
from rolepath_models import install_rolepath


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--arm",choices=ARMS,required=True); parser.add_argument("--device",default="cuda"); parser.add_argument("--jobs",type=int,default=4)
    args=parser.parse_args(); output=OUTPUT/"source"/args.arm
    if (output/"source_predictions.csv").is_file() and (output/"architecture.json").is_file():
        print("DIRECT_C43_ROLEPATH_FACTORIAL_RESUME arm="+args.arm,flush=True); return
    sys.path.insert(0,str(PRIOR_SCRIPT))
    from common import ENDPOINT_SOURCE, SOURCE_A0_CACHE
    from model_adapter import install_a0, prepare_imports
    from nested_stage1 import NestedStage1
    prepare_imports()
    stage1_script=PRIOR_SCRIPT.parent/"care-p31-stage1-rf-selection-source-v1"
    if str(stage1_script) not in sys.path: sys.path.insert(0,str(stage1_script))
    from source_label_contract import install_source_label_contract
    import run_care_rolegraph_conditioned_dualstream_v4 as v4
    method=METHOD[args.arm]; install_source_label_contract(v4); install_a0(v4,method)
    install_rolepath(v4,args.arm); install_direct(v4)
    nested=NestedStage1("SVM",OUTPUT/"audits"/f"{args.arm}_nested.json",args.jobs); v4.inner_oof_global24=nested.inner_oof
    argv=[sys.argv[0],"--previous-result",str(PRIOR_OUTPUT/"references/SVM"),"--cache",str(ENDPOINT_SOURCE),
          "--region-cache",str(SOURCE_A0_CACHE),"--variants","RGCondAttn","--seeds","0,1,2","--max-delta","2.25",
          "--device",args.device,"--output",str(output)]
    argv += ["--keep-weight","1","--fix-weight","1","--keep-consistency","0"]
    sys.argv=argv; v4.main()
    if len(nested.calls)!=5: raise RuntimeError("BLOCKED_BY_CONTRACT: expected five nested Stage-I calls")
    path=output/"architecture.json"; value=json.loads(path.read_text(encoding="utf-8")); value.update({
        "arm":args.arm,"method":method,"stage1":"P31 calibrated SVM","router":"SVM MeanOpposition Top25%",
        "bounded_correction_removed":True,"future_rows_loaded":0,"same_parameter_set":True,
        "semantics":"direct Stage-II RoleGraph query/readout 2x2 factorial; fixed non-trainable path switches"})
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("DIRECT_C43_ROLEPATH_FACTORIAL_ARM_DONE arm="+args.arm,flush=True)


if __name__=="__main__":main()
