#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent; PROJECT=HERE.parent.parent if HERE.parent.name=="scripts" else HERE.parent; SCRIPTS=PROJECT/"scripts" if (PROJECT/"scripts").is_dir() else PROJECT; BASE=SCRIPTS/"care-di25-graphglobal-path-source-v1"
for p in (HERE,BASE,SCRIPTS,SCRIPTS/"care-rolegraph-conditioned-v4-script",SCRIPTS/"care-rgcondattn-dedup-bounded-source-v1",SCRIPTS/"care-malben-host-v2"): sys.path.insert(0,str(p))
from compact_background import make_fit
from compact_models import ARMS,make_compact_class
from source_label_contract import install_source_label_contract
import run_care_rolegraph_conditioned_dualstream_v4 as v4

ROOT=Path("artifacts/care-di25-compact43-relationquery-source-v1")
METHOD={"C43_COMPACT":"DI25__Compact43__Anomaly__QQuery__Bounded","C43_RELATION_QUERY":"DI25__Compact43__Anomaly__GraphGlobalRelationQuery__Bounded"}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--arm",required=True,choices=tuple(ARMS)); p.add_argument("--seeds",default="0,1,2"); p.add_argument("--device",default="auto"); a=p.parse_args()
    install_source_label_contract(v4); v4.MODEL_VARIANTS=("RGCondAttn",); v4.METHOD_FOR_VARIANT={"RGCondAttn":METHOD[a.arm]}; v4.AUGMENTED_REGION_DIM=78
    v4.RoleGraphConditionedEvidence=make_compact_class(v4,a.arm); v4.v3.fit_scaled_dualproto_bags=make_fit(v4.v3)
    out=ROOT/"arms"/a.arm; out.mkdir(parents=True,exist_ok=True)
    sys.argv=[sys.argv[0],"--previous-result",str(ROOT/"preflight/di25_reference"),"--cache","artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz","--region-cache",str(ROOT/"cache/compact43_relation_source.npz"),"--variants","RGCondAttn","--seeds",a.seeds,"--max-delta","2.25","--device",a.device,"--output",str(out)]
    v4.main()
    with (out/"fit_audit.csv").open("r",encoding="utf-8-sig",newline="") as f: fit=list(csv.DictReader(f))
    counts={int(float(r["parameter_n"])) for r in fit}
    if len(counts)!=1 or {int(float(r["shared_prototype_n"])) for r in fit}!={32} or {int(float(r["benign_prototype_n"])) for r in fit}!={24}: raise RuntimeError("BLOCKED_BY_CONTRACT: capacity audit")
    contract={"arm":a.arm,"method":METHOD[a.arm],"source_only":True,"compact_host_width":43,"duplicate_role_columns":0,"model_region_width":78,"relation_query_width":31,"relation_enters_attention":bool(ARMS[a.arm]),"relation_enters_region_value":False,"relation_enters_background_distance":False,"relation_enters_delta":False,"background_role_distance_weight":2.0,"parameter_count":next(iter(counts)),"future_rows_loaded":0,"ego_consumed":False,"bounded_formula":"sigmoid(logit(p_stage1)+2.25*tanh(raw_delta))"}
    (out/"contract.json").write_text(json.dumps(contract,indent=2)+"\n"); print("COMPACT43_RELATIONQUERY_ARM_DONE",a.arm,flush=True)


if __name__=="__main__": main()
