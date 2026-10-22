#!/usr/bin/env python3
from pathlib import Path
import csv,json,numpy as np
ROOT=Path("artifacts/care-role-conditioned-api-v1")
REQ=(Path("artifacts/care-p31rf-c43-haig-host-ablation-source-v1/arms/A0_ROLE_ONLY/source_predictions.csv"),Path("artifacts/care-p31rf-c43-haig-host-ablation-source-v1/arms/A3_ROLE_PLUS_H3/source_predictions.csv"),Path("artifacts/care-p31-stage1-rf-selection-source-v1/preflight/C_RF_RFROUTE/source_predictions.csv"),Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz"),Path("artifacts/care-di25-compact43-relationquery-source-v1/cache/compact43_relation_source.npz"),Path("artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz"))
def rows(p):
    with p.open(encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def main():
    missing=[str(p) for p in REQ if not p.exists() or p.stat().st_size==0]
    if missing:raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty inputs:\n- "+"\n- ".join(missing))
    ref=rows(REQ[2])
    if len(ref)!=799 or sum(int(float(r["A25__routed"])) for r in ref)!=200 or {int(float(r["fold"])) for r in ref}!={1,2,3,4,5}:raise RuntimeError("BLOCKED_BY_CONTRACT: P31-RF/fold/route contract changed")
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/"arms").mkdir(exist_ok=True);(ROOT/"logs").mkdir(exist_ok=True);(ROOT/"audits").mkdir(exist_ok=True)
    protocol={"experiment":"Role-conditioned API refinement v1","source_rows":799,"year":2017,"fixed_outer_folds":5,"stage1":"P31-RF","router":"RF-anchored MeanOpposition DI25 Top25%","route_rows":200,"threshold":.5,"max_delta":2.25,"arms":["R0_LOCKED_ROLE","R1_UNCONDITIONAL_API","R2_FIXED_ROLE_API","R3_LEARNED_ROLE_API","R4_SHUFFLED_WITHIN_ROLE"],"R2_alpha":.25,"R3_alpha_max":.30,"api_embedding_dim":16,"vocabulary":"outer-train only, min APK DF 2, max 2048","validation_unseen":"UNK_API_WITHIN_ROLE","future_rows_loaded":0,"future_allowed_for_selection":False}
    (ROOT/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n");(ROOT/"preflight.json").write_text(json.dumps({"status":"PASS","source_rows":799,"route_rows":200,"folds":[1,2,3,4,5],"required_inputs":[str(p) for p in REQ],"future_rows_loaded":0},indent=2)+"\n");(ROOT/"feature_schema.json").write_text(json.dumps({"base_host":"Compact43 with four existing background quantities","localization":"RoleGraph plus Relation31 unchanged","api_identity":"exact stable SHA-256-derived canonical-signature identity from endpoint cache","api_role_pool":"31 roles x SVD16, L2-normalized Host-role rows","R2":"SelfRole31 + 0.25*tanh(role-API residual)","R3_R4":"SelfRole31 + [0.30*sigmoid(role gate)]*tanh(role-API residual)","background_use":"API residual excluded"},indent=2)+"\n");print("ROLE_API_PREFLIGHT_PASS");print("source_rows=799 route=200 future_rows_loaded=0")
if __name__=="__main__":main()
