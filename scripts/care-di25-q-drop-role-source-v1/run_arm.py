#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent.parent;DEP=PROJECT/"scripts/care-di25-graphglobal-path-source-v1"
for p in (HERE,DEP,PROJECT/"scripts",PROJECT/"scripts/care-rolegraph-conditioned-v4-script",PROJECT/"scripts/care-rgcondattn-dedup-bounded-source-v1",PROJECT/"scripts/care-malben-host-v2"):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from experiment_contract import MAX_DELTA,build_di_reference
from source_label_contract import install_source_label_contract
from drop_role_model import ARM,make_model
import run_care_rolegraph_conditioned_dualstream_v4 as v4

def main():
    p=argparse.ArgumentParser();p.add_argument("--device",default="cuda");p.add_argument("--seeds",default="0,1,2");p.add_argument("--output",type=Path,default=Path("artifacts/care-di25-q-drop-role-source-v1"));a=p.parse_args()
    root=a.output;out=root/"arm";out.mkdir(parents=True,exist_ok=True);contract=build_di_reference(PROJECT,root);install_source_label_contract(v4)
    method="DI25__Host__Anomaly__Q_DROP_ROLE_DIRECT__Bounded";v4.MODEL_VARIANTS=("RGCondAttn",);v4.METHOD_FOR_VARIANT={"RGCondAttn":method};v4.RoleGraphConditionedEvidence=make_model(v4)
    sys.argv=[sys.argv[0],"--previous-result",str(root/"preflight/di25_reference"),"--cache","artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz","--region-cache","artifacts/care-source-local-assets-v2/regions_2017.npz","--variants","RGCondAttn","--seeds",a.seeds,"--max-delta",str(MAX_DELTA),"--device",a.device,"--output",str(out)]
    v4.main()
    with (out/"fit_audit.csv").open("r",encoding="utf-8-sig",newline="") as h:fit=list(csv.DictReader(h))
    counts={int(float(r["parameter_n"])) for r in fit}
    if counts!={33036}:raise RuntimeError("BLOCKED_BY_CONTRACT: parameter count "+repr(counts))
    (out/"architecture.json").write_text(json.dumps({"arm":ARM,"parameter_count":33036,"correction_width":113,"active_dimensions":81,"role_token_to_attention":True,"graphglobal_to_attention":True,"role_token_to_delta":False,"global24_to_delta":True,"stage1_logit_to_delta":True,"host_to_delta":True,"route_DI":contract["route_DI"],"future_rows_loaded":0,"ego_consumed":False},indent=2)+"\n",encoding="utf-8")
    print("Q_DROP_ROLE_SOURCE_ARM_DONE")
if __name__=="__main__":main()
