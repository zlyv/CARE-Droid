#!/usr/bin/env python3
from common import *
def main():
    before=read_json(OUTPUT/"baseline_asset_manifest_before.json"); after=[]; changed=[]; missing=[]
    for row in before:
        path=Path(row["absolute_path"])
        if not path.is_file():missing.append(str(path));continue
        now=asset(path,row["purpose"]);after.append(now)
        if now["sha256"]!=row["sha256"]:changed.append(str(path))
    dump_json(OUTPUT/"baseline_asset_manifest_after.json",after); result={"status":"PASS" if not changed and not missing else "FAIL","asset_count":len(before),"changed":changed,"missing":missing};dump_json(OUTPUT/"baseline_integrity_check.json",result)
    if result["status"]!="PASS":raise RuntimeError("BLOCKED_BY_CONTRACT: formal assets changed "+repr(result))
    print("TBFRS_BASELINE_INTEGRITY_PASS assets=%d"%len(before),flush=True)
if __name__=="__main__":main()
