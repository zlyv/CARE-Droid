#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent.parent if HERE.parent.name=="scripts" else HERE.parent;SCRIPTS=PROJECT/"scripts" if (PROJECT/"scripts").is_dir() else PROJECT
BASE=SCRIPTS/"care-di25-graphglobal-path-source-v1";COMPACT=SCRIPTS/"care-di25-compact43-relationquery-source-v1"
for path in (HERE,COMPACT,BASE,SCRIPTS,SCRIPTS/"care-rolegraph-conditioned-v4-script",SCRIPTS/"care-rgcondattn-dedup-bounded-source-v1",SCRIPTS/"care-malben-host-v2"):sys.path.insert(0,str(path))
from compact_background import make_fit
from compact_models import make_compact_class
from routed_training import install
from source_label_contract import install_source_label_contract
import run_care_rolegraph_conditioned_dualstream_v4 as v4

ROOT=Path("artifacts/care-di25-c43-routed-training-source-v1")
METHOD={"T1_ROUTED_LOSS":"DI25__C43RQ__RoutedLoss__Bounded","T2_FULLY_ROUTED":"DI25__C43RQ__FullyRoutedTrain__Bounded"}

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--arm",required=True,choices=tuple(METHOD));parser.add_argument("--device",default="auto");parser.add_argument("--seeds",default="0,1,2");args=parser.parse_args()
    install_source_label_contract(v4);v4.MODEL_VARIANTS=("RGCondAttn",);v4.METHOD_FOR_VARIANT={"RGCondAttn":METHOD[args.arm]};v4.AUGMENTED_REGION_DIM=78
    v4.RoleGraphConditionedEvidence=make_compact_class(v4,"C43_RELATION_QUERY");v4.v3.fit_scaled_dualproto_bags=make_fit(v4.v3);state=install(v4,args.arm)
    out=ROOT/"arms"/args.arm;out.mkdir(parents=True,exist_ok=True)
    sys.argv=[sys.argv[0],"--previous-result",str(ROOT/"preflight/di25_reference"),"--cache","artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz","--region-cache","artifacts/care-di25-compact43-relationquery-source-v1/cache/compact43_relation_source.npz","--variants","RGCondAttn","--seeds",args.seeds,"--max-delta","2.25","--device",args.device,"--output",str(out)]
    v4.main()
    if any(np.any(~np.isfinite(values)) for values in state["seed_outputs"].values()):raise RuntimeError("BLOCKED_BY_CONTRACT: incomplete per-seed OOF")
    seed_rows=[]
    for i,sid in enumerate(state["ids"]):
        row={"sha256":sid,"fold":int(state["fold"][i]),"route_DI":int(state["route"][i]),"p_stage1":float(state["base"][i])}
        for offset,expert in state["seed_outputs"].items():row[f"p_expert_seed_{offset}"]=float(expert[i]);row[f"p_final_seed_{offset}"]=float(expert[i] if state["route"][i] else state["base"][i])
        seed_rows.append(row)
    with (out/"per_seed_predictions.csv").open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(seed_rows[0]));writer.writeheader();writer.writerows(seed_rows)
    with (out/"fit_audit.csv").open("r",encoding="utf-8-sig",newline="") as handle:rows=list(csv.DictReader(handle))
    if len(rows)!=15 or {int(float(r["optimization_train_n"])) for r in rows}!={160} or {int(float(r["parameter_n"])) for r in rows}!={32292}:
        raise RuntimeError("BLOCKED_BY_CONTRACT: routed training/capacity audit failed")
    contract={"arm":args.arm,"method":METHOD[args.arm],"source_only":True,"outer_train_n":"639 (folds 1-4) or 640 (fold 5)","routed_outer_train_n":160,"selected_epochs_reused_from_locked_T0":True,"router_applied_to_neural_loss":True,"background_scope":rows[0]["background_scope"],"stage2_scaler_scope":rows[0]["stage2_scaler_scope"],"validation_predictions_generated_for_all_APKs":True,"final_correction_applied_only_to_DI25":True,"per_seed_predictions_persisted":True,"parameter_count":32292,"future_rows_loaded":0,"ego_consumed":False}
    (out/"contract.json").write_text(json.dumps(contract,indent=2)+"\n");print("C43_ROUTED_TRAINING_ARM_DONE",args.arm,flush=True)
if __name__=="__main__":main()

