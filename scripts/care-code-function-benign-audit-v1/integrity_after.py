#!/usr/bin/env python3
from common import OUTPUT,asset,dump_json,read_json
def main():
    b=read_json(OUTPUT/"baseline_asset_manifest_before.json");a=[asset(x["path"],x["purpose"]) for x in b];changed=[x["path"] for x,y in zip(a,b) if x["sha256"]!=y["sha256"] or x["size"]!=y["size"]];dump_json(OUTPUT/"baseline_asset_manifest_after.json",a);dump_json(OUTPUT/"baseline_integrity_check.json",{"status":"PASS" if not changed else "FAIL","changed":changed})
    if changed:raise RuntimeError("BLOCKED_BY_CONTRACT: input changed")
    print("CODE_FUNCTION_BENIGN_INTEGRITY_PASS")
if __name__=="__main__":main()
