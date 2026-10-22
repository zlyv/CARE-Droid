#!/usr/bin/env python3
"""Train one strict-2017 Host-relative Q-Query arm."""
from __future__ import annotations

import argparse,json,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent.parent
BASE=PROJECT/"scripts/care-di25-graphglobal-path-source-v1"
for path in (HERE,BASE,PROJECT/"scripts",PROJECT/"scripts/care-rolegraph-conditioned-v4-script",PROJECT/"scripts/care-rgcondattn-dedup-bounded-source-v1",PROJECT/"scripts/care-malben-host-v2"):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
from experiment_contract import MAX_DELTA
from graphglobal_path_models import make_graphglobal_path_class
from source_label_contract import install_source_label_contract
from host_relative_features import ARMS
import run_care_rolegraph_conditioned_dualstream_v4 as v4

ROOT=Path("artifacts/care-di25-host-relative-source-v1")

def main():
    p=argparse.ArgumentParser();p.add_argument("--arm",required=True,choices=ARMS);p.add_argument("--seeds",default="0,1,2");p.add_argument("--device",default="auto");p.add_argument("--output",type=Path,default=ROOT);a=p.parse_args()
    if MAX_DELTA!=2.25:raise RuntimeError("BLOCKED_BY_CONTRACT: max_delta changed")
    contract_path=a.output/"preflight/di25_contract.json"
    if not contract_path.is_file():raise RuntimeError("BLOCKED_BY_CONTRACT: preflight DI25 contract missing")
    contract=json.loads(contract_path.read_text(encoding="utf-8"))
    if int(contract.get("source_rows",0))!=799 or int(contract.get("route_DI",0))!=200:raise RuntimeError("BLOCKED_BY_CONTRACT: preflight DI25 contract changed")
    install_source_label_contract(v4)
    method=f"DI25__HostRelative__Anomaly__{a.arm}__QQuery__Bounded";arm_dir=a.output/"arms"/a.arm;arm_dir.mkdir(parents=True,exist_ok=True)
    v4.MODEL_VARIANTS=("RGCondAttn",);v4.METHOD_FOR_VARIANT={"RGCondAttn":method};v4.RoleGraphConditionedEvidence=make_graphglobal_path_class(v4,"Q_QUERY_ONLY")
    sys.argv=[sys.argv[0],"--previous-result",str(a.output/"preflight/di25_reference"),"--cache","artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz","--region-cache",str(a.output/"caches"/(a.arm+"_regions_2017.npz")),"--variants","RGCondAttn","--seeds",a.seeds,"--max-delta",str(MAX_DELTA),"--device",a.device,"--output",str(arm_dir)]
    v4.main()
    path=arm_dir/"architecture.json";architecture=json.loads(path.read_text(encoding="utf-8"));architecture.update({"experiment":"CARE-Droid Host-relative source 2x2 v1","arm":a.arm,"method":method,"source_only":True,"future_rows_loaded":0,"ego_consumed":False,"host_width":74,"parameter_count_expected":33036,"graphglobal_to_attention":True,"graphglobal_to_delta_direct":False,"three_backgrounds_unchanged":["external_benign_distance","internal_dominant_background_distance","label_balanced_shared_commonness"],"payload_to_model_role_applied":True,"upstream_consumed":False,"downstream_consumed":False,"bounded_formula":"sigmoid(logit(p_stage1)+2.25*tanh(raw_delta))","route_DI":contract["route_DI"]})
    path.write_text(json.dumps(architecture,indent=2)+"\n",encoding="utf-8");print("HOST_RELATIVE_ARM_DONE",a.arm,flush=True)
if __name__=="__main__":main()
