#!/usr/bin/env python3
from common import OUTPUT,asset,dump_json,read_json
def main():
    before=read_json(OUTPUT/"baseline_asset_manifest_before.json");after=[asset(x["path"],x["purpose"]) for x in before];changed=[a["path"] for a,b in zip(after,before) if a["sha256"]!=b["sha256"] or a["size"]!=b["size"]]
    dump_json(OUTPUT/"baseline_asset_manifest_after.json",after);dump_json(OUTPUT/"baseline_integrity_check.json",{"status":"PASS" if not changed else "FAIL","changed":changed})
    if changed:raise RuntimeError("BLOCKED_BY_CONTRACT: formal asset changed")
    print("FCBE_BASELINE_INTEGRITY_PASS")
if __name__=="__main__":main()
