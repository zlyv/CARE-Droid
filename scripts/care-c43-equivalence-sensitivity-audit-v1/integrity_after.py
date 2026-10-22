#!/usr/bin/env python3
from audit_common import *

def main():
    before_path=OUTPUT/"baseline_asset_manifest_before.json"
    before=json.loads(before_path.read_text(encoding="utf-8"))["assets"]
    after=[asset(Path(row["absolute_path"]),row["purpose"]) for row in before]
    dump_json(OUTPUT/"baseline_asset_manifest_after.json",{"git_used":False,"assets":after})
    a={r["absolute_path"]:r for r in before}; b={r["absolute_path"]:r for r in after}
    changed=[{"path":p,"before":a.get(p),"after":b.get(p)} for p in sorted(set(a)|set(b)) if a.get(p)!=b.get(p)]
    dump_json(OUTPUT/"baseline_integrity_check.json",{"status":"PASS" if not changed else "FAIL","changed":changed,"asset_count":len(after)})
    if changed: raise RuntimeError("BLOCKED_BY_CONTRACT: protected assets changed")
    print("C43_EQUIVALENCE_SENSITIVITY_INTEGRITY_PASS",flush=True)

if __name__=="__main__":main()
