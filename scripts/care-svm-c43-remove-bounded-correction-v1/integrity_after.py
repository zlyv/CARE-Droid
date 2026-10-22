#!/usr/bin/env python3
from pathlib import Path
from ablation_common import OUTPUT, asset, dump_json, read_json


def main():
    before=read_json(OUTPUT/"baseline_asset_manifest_before.json"); after=[]; changed=[]
    for row in before:
        current=asset(Path(row["absolute_path"]),row["purpose"]); after.append(current)
        if current["sha256"]!=row["sha256"] or current["size_bytes"]!=row["size_bytes"]: changed.append(row["absolute_path"])
    status="PASS" if not changed else "FAIL"; dump_json(OUTPUT/"baseline_asset_manifest_after.json",after); dump_json(OUTPUT/"baseline_integrity_check.json",{"status":status,"changed":changed,"asset_count":len(after)})
    decision=read_json(OUTPUT/"final_decision.json"); decision["baseline_integrity"]=status; dump_json(OUTPUT/"final_decision.json",decision)
    if changed: raise RuntimeError("BLOCKED_BY_CONTRACT: formal assets changed")
    print("SVM_C43_REMOVE_BOUNDED_INTEGRITY_PASS",flush=True)


if __name__=="__main__":main()
