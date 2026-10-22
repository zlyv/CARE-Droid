#!/usr/bin/env python3
from common import FORMAL,INVENTORY,OUTPUT,P31,PARENT_PROBE,asset,dump_json,read_csv,read_json
def main():
    for p in (FORMAL,P31,INVENTORY,PARENT_PROBE):
        if not p.is_file():raise RuntimeError("BLOCKED_BY_CONTRACT: missing "+str(p))
    formal=read_csv(FORMAL);inv=read_csv(INVENTORY);parent=read_json(PARENT_PROBE)
    if len(formal)!=799 or len(inv)!=799 or sum(int(float(r["A25__routed"])) for r in formal)!=200:raise RuntimeError("BLOCKED_BY_CONTRACT: cohort/route changed")
    observed=float(parent["pooled"]["code_FP_vs_TP_AUC"])
    if abs(observed-.6938472942920682)>1e-12:raise RuntimeError("BLOCKED_BY_CONTRACT: parent code-only result changed")
    assets=[asset(FORMAL,"formal routed source predictions"),asset(P31,"frozen P31"),asset(INVENTORY,"static code-family inventory"),asset(PARENT_PROBE,"parent probe")]
    dump_json(OUTPUT/"baseline_asset_manifest_before.json",assets);dump_json(OUTPUT/"preflight.json",{"status":"PASS","source_rows":799,"fixed_route":200,"parent_code_auc":observed,"future_rows_loaded":0})
    print("CODE_FUNCTION_BENIGN_AUDIT_PREFLIGHT_PASS parent_code_auc=0.693847 route=200")
if __name__=="__main__":main()
