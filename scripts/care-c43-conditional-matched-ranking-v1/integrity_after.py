#!/usr/bin/env python3
from common import OUTPUT, dump_json, file_record, read_json

def main():
    before=read_json(OUTPUT/"baseline_asset_manifest_before.json")["assets"]
    after=[file_record(row["absolute_path"],row["purpose"]) for row in before]
    left={r["absolute_path"]:r["sha256"] for r in before}; right={r["absolute_path"]:r["sha256"] for r in after}
    changed=[p for p in sorted(set(left)|set(right)) if left.get(p)!=right.get(p)]
    status="PASS" if not changed else "FAIL"
    dump_json(OUTPUT/"baseline_asset_manifest_after.json",{"assets":after})
    dump_json(OUTPUT/"baseline_integrity_check.json",{"status":status,"asset_count":len(after),"changed":changed})
    print("C43_CONDITIONAL_MATCHED_INTEGRITY_"+status,flush=True)
    if changed: raise RuntimeError("BLOCKED_BY_CONTRACT: frozen assets changed")
if __name__=="__main__": main()
