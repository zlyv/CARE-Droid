#!/usr/bin/env python3
from __future__ import annotations
from common import (OUTPUT, discover_assets, dump_json, ensure_output_boundary,
                    load_apk_paths, norm, read_csv)

def main():
    ensure_output_boundary(); OUTPUT.mkdir(parents=True,exist_ok=True)
    assets=discover_assets(); rows=read_csv(assets["ceiling"])
    required={"sha256","year","label","stage1_wrong","current_wrong","family_plus_expert_unanimous_wrong",
              "LR_prob","RF_prob","XGB_prob","SVM_prob","expert_prob"}
    missing=sorted(required-set(rows[0] if rows else {}))
    if len(rows)!=2788 or missing:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: ceiling table rows={len(rows)} missing={missing}")
    consensus=[r for r in rows if int(float(r["family_plus_expert_unanimous_wrong"]))]
    by_year={year:sum(int(float(r["family_plus_expert_unanimous_wrong"])) for r in rows if int(float(r["year"]))==year) for year in range(2017,2022)}
    if by_year.get(2017)!=25 or sum(by_year.values())!=93:
        raise RuntimeError("BLOCKED_BY_CONTRACT: frozen consensus-error counts changed "+repr(by_year))
    wanted={norm(r["sha256"]) for r in rows}; apk_map,audit=load_apk_paths(wanted)
    dump_json(OUTPUT/"preflight.json",{"status":"PASS" if set(apk_map)==wanted else "FAIL","rows":len(rows),
        "consensus_errors":len(consensus),"consensus_errors_by_year":by_year,"apk_paths":len(apk_map),
        "future_used_for_selection":False,"verified_fingerprint":str(assets["old_fingerprint"]),"path_sources":audit,
        "missing_apks":sorted(wanted-set(apk_map))[:100]})
    if set(apk_map)!=wanted:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: APK paths {len(apk_map)}/{len(wanted)}")
    print("CONSENSUS_ERROR_LINEAGE_PREFLIGHT_PASS rows=2788 consensus_errors=93 apk_paths=2788",flush=True)
if __name__=="__main__":main()
