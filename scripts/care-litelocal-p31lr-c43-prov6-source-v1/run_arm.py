#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent.parent;SCRIPTS=PROJECT/"scripts"
SOURCE_VENDOR=SCRIPTS/"care-stage2only-anomaly-v1/vendor"
for p in (HERE,SCRIPTS/"care-di25-graphglobal-path-source-v1",SOURCE_VENDOR,SCRIPTS/"care-rolegraph-conditioned-v4-script",SCRIPTS/"care-rgcondattn-dedup-bounded-source-v1",SCRIPTS/"care-malben-host-v2",SCRIPTS):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from source_label_contract import install_source_label_contract
import run_care_rolegraph_conditioned_dualstream_v4 as v4
from nested_lr import NestedLR
from prov6_background import make_fit
from prov6_models import make_class

ROOT=Path("artifacts/care-litelocal-p31lr-c43-prov6-source-v1")
ARMS={"P0_ZERO6":"P31LR__Opp25__C43Prov6Zero__Bounded","P1_PROV6":"P31LR__Opp25__C43Prov6Real__Bounded","P2_SHUFFLED_PROV6":"P31LR__Opp25__C43Prov6Shuffled__Bounded"}
def main():
    p=argparse.ArgumentParser();p.add_argument("--arm",choices=ARMS,required=True);p.add_argument("--device",default="auto");p.add_argument("--seeds",default="0,1,2");a=p.parse_args();install_source_label_contract(v4)
    method=ARMS[a.arm];v4.MODEL_VARIANTS=("RGCondAttn",);v4.METHOD_FOR_VARIANT={"RGCondAttn":method};v4.AUGMENTED_REGION_DIM=84;v4.RoleGraphConditionedEvidence=make_class(v4);v4.v3.fit_scaled_dualproto_bags=make_fit(v4.v3)
    nested=NestedLR();v4.inner_oof_global24=nested.inner_oof;out=ROOT/"arms"/a.arm;out.mkdir(parents=True,exist_ok=True)
    sys.argv=[sys.argv[0],"--previous-result",str(ROOT/"preflight/lr_reference"),"--cache","artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz","--region-cache",str(ROOT/"cache"/(a.arm+".npz")),"--variants","RGCondAttn","--seeds",a.seeds,"--max-delta","2.25","--device",a.device,"--output",str(out)]
    v4.main()
    if len(nested.calls)!=5:raise RuntimeError("BLOCKED_BY_CONTRACT: expected five nested LR calls")
    path=out/"architecture.json";arch=json.loads(path.read_text());arch.update({"experiment":"P31-LR + LR-anchored committee router + C43-RQ Prov6","arm":a.arm,"stage1":"calibrated P31 Logistic Regression only","committee_role":"routing only","outer_train_stage1":"nested inner-OOF calibrated P31-LR","provenance_width":6,"provenance_enters_host_encoder":True,"relation_enters_attention":True,"relation_enters_delta":False,"scalar_bounded_head_unchanged":True,"max_delta":2.25,"future_rows_loaded":0,"ego_consumed":False});path.write_text(json.dumps(arch,indent=2)+"\n")
    print("P31LR_PROV6_ARM_DONE",a.arm)
if __name__=="__main__":main()
