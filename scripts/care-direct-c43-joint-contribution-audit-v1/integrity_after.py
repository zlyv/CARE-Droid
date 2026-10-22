#!/usr/bin/env python3
import json
from audit_common import *
def main():
    before=json.loads((OUTPUT/"baseline_asset_manifest_before.json").read_text(encoding="utf-8"))["assets"]
    after=[asset(Path(r["absolute_path"]),r["purpose"]) for r in before]; changed=[{"before":a,"after":b} for a,b in zip(before,after) if a["sha256"]!=b["sha256"]]
    dump_json(OUTPUT/"baseline_asset_manifest_after.json",{"assets":after}); dump_json(OUTPUT/"baseline_integrity_check.json",{"status":"PASS" if not changed else "FAIL","changed":changed,"asset_count":len(after)})
    if changed:raise RuntimeError("BLOCKED_BY_CONTRACT: formal asset changed")
    decision=json.loads((OUTPUT/"final_decision.json").read_text(encoding="utf-8")); decision["formal_integrity"]="PASS"; dump_json(OUTPUT/"final_decision.json",decision)
    print("DIRECT_C43_JOINT_AUDIT_INTEGRITY_PASS",flush=True)
if __name__=="__main__":main()
