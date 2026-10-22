#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;SCRIPTS=HERE.parent
for p in (HERE,SCRIPTS/"care-p31-stage1-rf-selection-source-v1",SCRIPTS/"care-di25-graphglobal-path-source-v1",SCRIPTS/"care-stage2only-anomaly-v1/vendor",SCRIPTS/"care-rolegraph-conditioned-v4-script",SCRIPTS/"care-rgcondattn-dedup-bounded-source-v1",SCRIPTS/"care-malben-host-v2",SCRIPTS):sys.path.insert(0,str(p)) if str(p) not in sys.path else None
from source_label_contract import install_source_label_contract
import run_care_rolegraph_conditioned_dualstream_v4 as v4
from nested_stage1 import NestedStage1
from haig_background import make_fit
from haig_models import make_class
ROOT=Path("artifacts/care-p31rf-c43-haig-host-ablation-source-v1")
ARMS=("A0_ROLE_ONLY","A1_API_NO_ROLE","A2_H3_NO_ROLE","A3_ROLE_PLUS_H3","A4_H3_SHUFFLE_NO_ROLE")
def main():
    p=argparse.ArgumentParser();p.add_argument("--arm",choices=ARMS,required=True);p.add_argument("--device",default="cuda");p.add_argument("--seeds",default="0,1,2");a=p.parse_args();install_source_label_contract(v4);method="P31RF__RFRoute__"+a.arm+"__Bounded";v4.MODEL_VARIANTS=("RGCondAttn",);v4.METHOD_FOR_VARIANT={"RGCondAttn":method};v4.AUGMENTED_REGION_DIM=142;v4.RoleGraphConditionedEvidence=make_class(v4);v4.v3.fit_scaled_dualproto_bags=make_fit(v4.v3);nested=NestedStage1("RF",ROOT/"audits"/(a.arm+"_nested.json"),4);v4.inner_oof_global24=nested.inner_oof;out=ROOT/"arms"/a.arm;out.mkdir(parents=True,exist_ok=True);sys.argv=[sys.argv[0],"--previous-result","artifacts/care-p31-stage1-rf-selection-source-v1/preflight/C_RF_RFROUTE","--cache","artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz","--region-cache",str(ROOT/"cache"/(a.arm+".npz")),"--variants","RGCondAttn","--seeds",a.seeds,"--max-delta","2.25","--device",a.device,"--output",str(out)];v4.main()
    if len(nested.calls)!=5:raise RuntimeError("BLOCKED_BY_CONTRACT: expected five nested RF calls")
    q=out/"architecture.json";d=json.loads(q.read_text());d.update({"arm":a.arm,"stage1":"P31 calibrated RF only","router":"RF-anchored MeanOpposition Top25%","role_removed_scope":"Host content only in A1/A2/A4","RoleGraph_and_Relation31_retained_for_localization":True,"HAIG_relation":"direct Host-call interaction; not data flow","max_delta":2.25,"future_rows_loaded":0});q.write_text(json.dumps(d,indent=2)+"\n");print("P31RF_C43_HAIG_HOST_ARM_DONE",a.arm)
if __name__=="__main__":main()

