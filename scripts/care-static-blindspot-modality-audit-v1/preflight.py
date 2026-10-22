#!/usr/bin/env python3
from __future__ import annotations
from common import (OUTPUT,discover_assets,dump_json,ensure_output_boundary,
                    load_apk_paths,norm,read_csv)

def main():
    ensure_output_boundary();OUTPUT.mkdir(parents=True,exist_ok=True);assets=discover_assets();rows=read_csv(assets["ceiling"])
    required={"sha256","year","fold","label","stage1_prob","expert_prob","LR_prob","RF_prob","XGB_prob","SVM_prob","stage1_wrong","current_wrong","family_plus_expert_unanimous_wrong"}
    missing=sorted(required-set(rows[0] if rows else {}));wanted={norm(r["sha256"]) for r in rows}
    if len(rows)!=2788 or missing:raise RuntimeError(f"BLOCKED_BY_CONTRACT: ceiling rows={len(rows)} missing={missing}")
    consensus=sum(int(float(r["family_plus_expert_unanimous_wrong"])) for r in rows)
    if consensus!=93:raise RuntimeError(f"BLOCKED_BY_CONTRACT: consensus count changed {consensus}")
    apk_map,audit=load_apk_paths(wanted);result={"status":"PASS" if set(apk_map)==wanted else "FAIL","rows":len(rows),"consensus_errors":consensus,"apk_paths":len(apk_map),"missing_apks":sorted(wanted-set(apk_map))[:100],"path_sources":audit,"future_used_for_selection":False}
    dump_json(OUTPUT/"preflight.json",result)
    if set(apk_map)!=wanted:raise RuntimeError(f"BLOCKED_BY_CONTRACT: APK paths {len(apk_map)}/{len(wanted)}")
    print("STATIC_BLINDSPOT_PREFLIGHT_PASS rows=2788 consensus=93 apk_paths=2788",flush=True)
if __name__=="__main__":main()
