#!/usr/bin/env python3
import json
from pathlib import Path
from common import *
def main():
    before=json.loads((OUTPUT/"baseline_asset_manifest_before.json").read_text(encoding="utf-8"));after=[];changed=[]
    for old in before["assets"]:
        now=asset(Path(old["absolute_path"]),old["purpose"]);after.append(now)
        if any(now[k]!=old[k] for k in ("size_bytes","mtime_ns","sha256")):changed.append({"before":old,"after":now})
    dump_json(OUTPUT/"baseline_asset_manifest_after.json",{"assets":after,"future_rows_loaded":0})
    result={"status":"PASS" if not changed else "FAIL","asset_count":len(after),"changed":changed}
    dump_json(OUTPUT/"baseline_integrity_check.json",result)
    if changed:raise RuntimeError("BLOCKED_BY_CONTRACT: formal assets changed")
    print("C43_STATIC_EVIDENCE_FIDELITY_INTEGRITY_PASS")
if __name__=="__main__":main()
