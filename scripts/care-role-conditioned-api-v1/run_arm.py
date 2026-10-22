#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;SCRIPTS=HERE.parent
for p in (HERE,SCRIPTS/"care-p31-stage1-rf-selection-source-v1",SCRIPTS/"care-di25-compact43-relationquery-source-v1",SCRIPTS/"care-di25-graphglobal-path-source-v1",SCRIPTS/"care-stage2only-anomaly-v1/vendor",SCRIPTS/"care-rolegraph-conditioned-v4-script",SCRIPTS/"care-rgcondattn-dedup-bounded-source-v1",SCRIPTS/"care-malben-host-v2",SCRIPTS):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from source_label_contract import install_source_label_contract
import run_care_rolegraph_conditioned_dualstream_v4 as v4
from nested_stage1 import NestedStage1
import role_api_background as bg
from role_api_models import ARMS,make_class,install_gate_capture,GATE_SNAPSHOTS
from role_api_core import write_csv
ROOT=Path("artifacts/care-role-conditioned-api-v1")
METHOD={a:"P31RF__RFRoute__"+a+"__Bounded" for a in ARMS}
def main():
    p=argparse.ArgumentParser();p.add_argument("--arm",choices=ARMS,required=True);p.add_argument("--device",default="cuda");p.add_argument("--seeds",default="0,1,2");a=p.parse_args()
    install_source_label_contract(v4);v4.MODEL_VARIANTS=("RGCondAttn",);v4.METHOD_FOR_VARIANT={"RGCondAttn":METHOD[a.arm]};v4.AUGMENTED_REGION_DIM=574;v4.RoleGraphConditionedEvidence=make_class(v4,a.arm);bg.configure(ROOT/"cache/role_api_registry.npz",a.arm);v4.v3.fit_scaled_dualproto_bags=bg.make_fit(v4.v3);install_gate_capture(v4,a.arm)
    nested=NestedStage1("RF",ROOT/"audits"/(a.arm+"_nested.json"),4);v4.inner_oof_global24=nested.inner_oof;out=ROOT/"arms"/a.arm;out.mkdir(parents=True,exist_ok=True)
    sys.argv=[sys.argv[0],"--previous-result","artifacts/care-p31-stage1-rf-selection-source-v1/preflight/C_RF_RFROUTE","--cache","artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz","--region-cache",str(ROOT/"cache/role_api_raw_source.npz"),"--variants","RGCondAttn","--seeds",a.seeds,"--max-delta","2.25","--device",a.device,"--output",str(out)];v4.main()
    if len(nested.calls)!=5:raise RuntimeError("BLOCKED_BY_CONTRACT: expected five nested RF calls")
    write_csv(out/"api_projection_audit.csv",bg.PROJECTION_AUDITS);write_csv(out/"apk_feature_audit.csv",bg.APK_FEATURE_AUDITS)
    full=[r["vocabulary_sha256"] for r in bg.PROJECTION_AUDITS if int(r["full_outer_fit"])]
    if len(full)!=15:raise RuntimeError(f"BLOCKED_BY_CONTRACT: expected 15 full-fit vocabulary audits, got {len(full)}")
    for fold in range(1,6):
        group=full[(fold-1)*3:fold*3]
        if len(set(group))!=1:raise RuntimeError(f"BLOCKED_BY_CONTRACT: seed-dependent vocabulary in fold {fold}")
        vh=group[0]
        payload=bg.VOCABULARIES[vh];(out/f"api_vocabulary_fold_{fold}.json").write_text(json.dumps({"fold":fold,"vocabulary_sha256":vh,"tokens":[{"canonical_signature_hash":token,"train_apk_df":payload["frequency"][token]} for token in payload["tokens"]]},indent=2)+"\n")
    gates=[]
    for index,row in enumerate(GATE_SNAPSHOTS):
        for role,value in enumerate(row["alpha"]):gates.append({"snapshot":index,"arm":a.arm,"role":role,"alpha":value})
    write_csv(out/"gate_snapshots.csv",gates)
    q=out/"architecture.json";d=json.loads(q.read_text());d.update({"arm":a.arm,"stage1":"P31 calibrated RF only","router":"RF-anchored MeanOpposition Top25%","host_content":"SelfRole31 + bounded role-conditioned exact-API residual","api_embedding":"outer-train vocabulary, role-specific UNK, train-only SVD16","api_residual_enters_background":False,"RoleGraph_and_Relation31_retained_for_localization":True,"max_delta":2.25,"future_rows_loaded":0});q.write_text(json.dumps(d,indent=2)+"\n");print("ROLE_API_ARM_DONE",a.arm)
if __name__=="__main__":main()
