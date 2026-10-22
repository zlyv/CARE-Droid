#!/usr/bin/env python3
from common import OUTPUT,asset,dump_json,read_json
def main():
    before=read_json(OUTPUT/"baseline_asset_manifest_before.json")["assets"];after=[asset(r["absolute_path"],r["purpose"]) for r in before]
    left={r["absolute_path"]:r["sha256"] for r in before};right={r["absolute_path"]:r["sha256"] for r in after};changed=[p for p in sorted(set(left)|set(right)) if left.get(p)!=right.get(p)]
    status="PASS" if not changed else "FAIL";dump_json(OUTPUT/"baseline_asset_manifest_after.json",{"assets":after});dump_json(OUTPUT/"baseline_integrity_check.json",{"status":status,"changed":changed,"asset_count":len(after)})
    print("DIRECT_C43_FRONTIER_INTEGRITY_"+status,flush=True)
    if changed:raise RuntimeError("BLOCKED_BY_CONTRACT: assets changed")
if __name__=="__main__":main()
