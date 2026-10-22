#!/usr/bin/env python3
from common import APK_PATH_CANDIDATES,FORMAL,OUTPUT,P31,asset,dump_json,read_csv,read_json
from taxonomy import schema
def main():
    path=next((p for p in APK_PATH_CANDIDATES if p.is_file()),None)
    if not FORMAL.is_file() or not P31.is_file() or path is None:raise RuntimeError("BLOCKED_BY_CONTRACT: missing formal/P31/APK assets")
    rows=read_csv(FORMAL);paths=read_json(path)
    if len(rows)!=799 or len(paths)!=799 or sum(int(float(r["A25__routed"])) for r in rows)!=200:raise RuntimeError("BLOCKED_BY_CONTRACT: frozen cohort/route changed")
    assets=[asset(FORMAL,"formal RF-DI25-C43 predictions"),asset(P31,"frozen P31 role prevalence"),asset(path,"frozen APK path map")]
    dump_json(OUTPUT/"baseline_asset_manifest_before.json",assets);dump_json(OUTPUT/"function_schema.json",schema());dump_json(OUTPUT/"preflight.json",{"status":"PASS","source_rows":799,"fixed_route":200,"analysis":"static_manifest_and_dex_only","future_rows_loaded":0})
    print("FCBE_PREFLIGHT_PASS source=799 route=200 static_only=true future_rows_loaded=0")
if __name__=="__main__":main()
