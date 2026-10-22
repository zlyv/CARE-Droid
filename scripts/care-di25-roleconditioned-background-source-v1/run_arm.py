#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent.parent;BASE=PROJECT/"scripts/care-di25-graphglobal-path-source-v1"
for p in (HERE,BASE,PROJECT/"scripts",PROJECT/"scripts/care-rolegraph-conditioned-v4-script",PROJECT/"scripts/care-rgcondattn-dedup-bounded-source-v1",PROJECT/"scripts/care-malben-host-v2"):
    sys.path.insert(0,str(p))
from semantic_background import ARMS,make_fit
from graphglobal_path_models import make_graphglobal_path_class
from source_label_contract import install_source_label_contract
import run_care_rolegraph_conditioned_dualstream_v4 as v4
ROOT=Path("artifacts/care-di25-roleconditioned-background-source-v1")
METHOD={"B1_HARD_ROLE":"DI25__Host__Anomaly__QQuery__HardRoleBackground__Bounded","B2_SOFT_ROLE":"DI25__Host__Anomaly__QQuery__SoftRoleBackground__Bounded"}
def main():
    p=argparse.ArgumentParser();p.add_argument("--arm",required=True,choices=ARMS);p.add_argument("--seeds",default="0,1,2");p.add_argument("--device",default="auto");a=p.parse_args()
    install_source_label_contract(v4);v4.MODEL_VARIANTS=("RGCondAttn",);v4.METHOD_FOR_VARIANT={"RGCondAttn":METHOD[a.arm]};v4.RoleGraphConditionedEvidence=make_graphglobal_path_class(v4,"Q_QUERY_ONLY");v4.v3.fit_scaled_dualproto_bags=make_fit(v4.v3,a.arm)
    out=ROOT/"arms"/a.arm;out.mkdir(parents=True,exist_ok=True)
    sys.argv=[sys.argv[0],"--previous-result",str(ROOT/"preflight/di25_reference"),"--cache","artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz","--region-cache",str(ROOT/"cache/regions_2017_role_sidecar.npz"),"--variants","RGCondAttn","--seeds",a.seeds,"--max-delta","2.25","--device",a.device,"--output",str(out)]
    v4.main()
    with (out/"fit_audit.csv").open("r",encoding="utf-8-sig",newline="") as f:fit=list(csv.DictReader(f))
    if {int(float(r["parameter_n"])) for r in fit}!={33036} or {int(float(r["shared_prototype_n"])) for r in fit}!={32} or {int(float(r["benign_prototype_n"])) for r in fit}!={24}:raise RuntimeError("BLOCKED_BY_CONTRACT: capacity changed")
    architecture={"arm":a.arm,"method":METHOD[a.arm],"source_only":True,"future_rows_loaded":0,"ego_consumed":False,"raw_host_width":74,"role_sidecar_width":31,"model_input_width_after_backgrounds":78,"parameter_count":33036,"shared_prototypes":32,"benign_prototypes":24,"prototype_capacity_matched_to_H0":True,"semantic_sidecar_enters_neural_model_directly":False,"three_backgrounds_semantically_conditioned":True,"hard_definition":"dominant endpoint role matches dominant prototype role" if a.arm=="B1_HARD_ROLE" else None,"soft_definition":"endpoint-role distribution dot prototype-role distribution; distance plus 1-compatibility" if a.arm=="B2_SOFT_ROLE" else None,"bounded_formula":"sigmoid(logit(p_stage1)+2.25*tanh(raw_delta))"}
    (out/"semantic_background_contract.json").write_text(json.dumps(architecture,indent=2)+"\n")
    print("ROLECONDITIONED_BACKGROUND_ARM_DONE",a.arm,flush=True)
if __name__=="__main__":main()
