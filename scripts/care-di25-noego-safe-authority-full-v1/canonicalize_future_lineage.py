#!/usr/bin/env python3
"""Recover the exact audited 1989-row future lineage from frozen asset SHAs."""
from __future__ import annotations

import argparse,csv,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

EXPECTED={2018:793,2019:399,2020:397,2021:400}
def norm(x):return str(x or "").strip().lower()
def rid(r):return norm(r.get("sha256") or r.get("sample_id") or r.get("apk_id"))
def main():
    p=argparse.ArgumentParser();p.add_argument("--lineage",type=Path,required=True);p.add_argument("--regions",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--audit",type=Path,required=True);a=p.parse_args()
    with np.load(a.regions,allow_pickle=False) as z:
        if "sha256" not in z.files:raise RuntimeError("BLOCKED_BY_CONTRACT: future regions lacks sha256")
        asset_ids=[norm(x) for x in z["sha256"].astype(str)]
    if len(asset_ids)!=1989 or len(set(asset_ids))!=1989:raise RuntimeError("BLOCKED_BY_CONTRACT: frozen future asset SHA contract changed")
    with a.lineage.open("r",encoding="utf-8-sig",newline="") as h:
        q=csv.DictReader(h);rows=list(q);fields=list(q.fieldnames or [])
    groups=defaultdict(list)
    for r in rows:groups[rid(r)].append(r)
    selected=[];duplicates={}
    for value in asset_ids:
        candidates=groups.get(value,[])
        if not candidates:raise RuntimeError("BLOCKED_BY_CONTRACT: lineage missing frozen future SHA "+value)
        first=candidates[0]
        if any(r!=first for r in candidates[1:]):raise RuntimeError("BLOCKED_BY_CONTRACT: conflicting lineage duplicate "+value)
        if len(candidates)>1:duplicates[value]=len(candidates)
        selected.append(first)
    years=Counter(int(r.get("year") or r.get("metadata_year")) for r in selected)
    if dict(years)!=EXPECTED:raise RuntimeError("BLOCKED_BY_CONTRACT: asset-anchored future year counts changed "+repr(dict(years)))
    labels=[]
    for r in selected:
        value=norm(r.get("label"))
        if value not in {"0","0.0","1","1.0","benign","goodware","malware","malicious"}:raise RuntimeError("BLOCKED_BY_CONTRACT: invalid future label")
        labels.append(value)
    outside=sorted(set(groups)-set(asset_ids)-{""})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(selected)
    audit={"raw_lineage":str(a.lineage.resolve()),"raw_rows":len(rows),"frozen_asset":str(a.regions.resolve()),"frozen_asset_rows":1989,
        "output":str(a.output.resolve()),"output_rows":len(selected),"year_counts":dict(years),"identical_duplicate_ids":duplicates,
        "raw_ids_outside_frozen_asset":outside,"authority":"SHA order/set from previously hard-gated regions_2018_2021.npz","passed":True}
    a.audit.parent.mkdir(parents=True,exist_ok=True);a.audit.write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    print(f"FUTURE_LINEAGE_CANONICALIZED raw={len(rows)} exact=1989 excluded={len(outside)} duplicates={len(duplicates)}")
if __name__=="__main__":main()
