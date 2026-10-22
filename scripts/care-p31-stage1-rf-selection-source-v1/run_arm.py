#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent; PROJECT=HERE.parent.parent; SCRIPTS=PROJECT/"scripts"
SOURCE_VENDOR=SCRIPTS/"care-stage2only-anomaly-v1/vendor"
for p in (HERE,SCRIPTS/"care-di25-graphglobal-path-source-v1",SOURCE_VENDOR,
          SCRIPTS/"care-litelocal-p31lr-c43-prov6-source-v1",
          SCRIPTS/"care-rolegraph-conditioned-v4-script",
          SCRIPTS/"care-rgcondattn-dedup-bounded-source-v1",
          SCRIPTS/"care-malben-host-v2",SCRIPTS):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from source_label_contract import install_source_label_contract
import run_care_rolegraph_conditioned_dualstream_v4 as v4
from nested_stage1 import NestedStage1
from prov6_background import make_fit
from prov6_models import make_class

ROOT=Path("artifacts/care-p31-stage1-rf-selection-source-v1")
ARMS={
 "B_RF_LRROUTE":("RF","P31RF__LRRoute__P0Zero6C43RQ__Bounded"),
 "C_RF_RFROUTE":("RF","P31RF__RFRoute__P0Zero6C43RQ__Bounded"),
 "D_MEAN_MEANROUTE":("MEAN","P31Mean__MeanRoute__P0Zero6C43RQ__Bounded"),
}

def main():
    p=argparse.ArgumentParser();p.add_argument("--arm",choices=ARMS,required=True);p.add_argument("--device",default="auto");p.add_argument("--seeds",default="0,1,2");p.add_argument("--jobs",type=int,default=4);a=p.parse_args()
    mode,method=ARMS[a.arm];install_source_label_contract(v4)
    v4.MODEL_VARIANTS=("RGCondAttn",);v4.METHOD_FOR_VARIANT={"RGCondAttn":method}
    v4.AUGMENTED_REGION_DIM=84;v4.RoleGraphConditionedEvidence=make_class(v4)
    v4.v3.fit_scaled_dualproto_bags=make_fit(v4.v3)
    nested=NestedStage1(mode,ROOT/"audits"/(a.arm+"_nested_calls.json"),a.jobs)
    v4.inner_oof_global24=nested.inner_oof
    out=ROOT/"arms"/a.arm;out.mkdir(parents=True,exist_ok=True)
    sys.argv=[sys.argv[0],"--previous-result",str(ROOT/"preflight"/a.arm),
        "--cache","artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz",
        "--region-cache","artifacts/care-litelocal-p31lr-c43-prov6-source-v1/cache/P0_ZERO6.npz",
        "--variants","RGCondAttn","--seeds",a.seeds,"--max-delta","2.25",
        "--device",a.device,"--output",str(out)]
    v4.main()
    if len(nested.calls)!=5:raise RuntimeError("BLOCKED_BY_CONTRACT: expected five nested calls")
    path=out/"architecture.json";arch=json.loads(path.read_text())
    arch.update({"experiment":"P31 Stage-I RF source selection","arm":a.arm,
        "stage1":mode,"router":a.arm.split("_",2)[-1],"committee_role":"routing",
        "stage2":"P0_ZERO6 C43-RQ","max_delta":2.25,"future_rows_loaded":0,
        "ego_consumed":False});path.write_text(json.dumps(arch,indent=2)+"\n")
    print("P31_RF_SELECTION_ARM_DONE",a.arm)

if __name__=="__main__":main()
