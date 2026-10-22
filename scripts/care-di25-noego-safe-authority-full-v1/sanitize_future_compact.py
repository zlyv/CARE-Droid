#!/usr/bin/env python3
"""Build an audited exact-lineage snapshot from a contaminated compact cache.

The source cache is never changed. Identical duplicates are collapsed with an
explicit audit; outside-lineage rows are excluded; conflicting or missing IDs
are deliberately left absent so the official resumable extractor rebuilds only
those IDs in the new output root.
"""
from __future__ import annotations

import argparse,csv,json,shutil
from collections import Counter,defaultdict
from pathlib import Path

EXPECTED={2018:793,2019:399,2020:397,2021:400}

def norm(x):return str(x or "").strip().lower()
def rid(r):return norm(r.get("sha256") or r.get("sample_id") or r.get("apk_id"))
def read(path):
    if not path.is_file() or path.stat().st_size==0:return [],[]
    with path.open("r",encoding="utf-8-sig",newline="") as h:
        q=csv.DictReader(h);return list(q),list(q.fieldnames or [])
def main():
    p=argparse.ArgumentParser();p.add_argument("--lineage",type=Path,required=True);p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    lineage,_=read(a.lineage);wanted=[rid(r) for r in lineage];years=Counter(int(r.get("year") or r.get("metadata_year")) for r in lineage)
    if len(wanted)!=1989 or len(set(wanted))!=1989 or dict(years)!=EXPECTED:raise RuntimeError("BLOCKED_BY_CONTRACT: future lineage changed")
    source_rows,fields=read(a.source/"features.csv")
    if not source_rows or "sha256" not in fields:raise RuntimeError("BLOCKED_BY_CONTRACT: source compact cache is unreadable")
    existing_rows,existing_fields=read(a.output/"features.csv")
    if existing_rows and existing_fields!=fields:raise RuntimeError("BLOCKED_BY_CONTRACT: partial clean cache schema differs")
    source_groups=defaultdict(list);existing_groups=defaultdict(list)
    for r in source_rows:source_groups[rid(r)].append(r)
    for r in existing_rows:existing_groups[rid(r)].append(r)
    wanted_set=set(wanted);selected={};identical={};conflicting={};non_ok=[]
    for value in wanted:
        # A previously extracted row in the clean target has priority on resume.
        pool=existing_groups.get(value,[]) or source_groups.get(value,[])
        candidates=[r for r in pool if str(r.get("status","OK")).strip()=="OK"]
        if not candidates:continue
        canonical=candidates[0]
        if all(r==canonical for r in candidates[1:]):
            selected[value]=canonical
            if len(candidates)>1:identical[value]=len(candidates)
        else:
            conflicting[value]=len(candidates)
    for r in source_rows:
        if rid(r) in wanted_set and str(r.get("status","OK")).strip()!="OK":non_ok.append(rid(r))
    outside=sorted((set(source_groups)|set(existing_groups))-wanted_set-{""});missing=sorted(wanted_set-set(selected))
    a.output.mkdir(parents=True,exist_ok=True)
    tmp=a.output/"features.csv.tmp"
    with tmp.open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(selected[v] for v in wanted if v in selected)
    tmp.replace(a.output/"features.csv")
    schema=a.source/"feature_schema.json"
    if not schema.is_file():raise RuntimeError("BLOCKED_BY_CONTRACT: source cache lacks feature_schema.json")
    shutil.copy2(schema,a.output/"feature_schema.json")
    audit={"source_cache":str(a.source.resolve()),"output_cache":str(a.output.resolve()),"lineage_rows":1989,
        "source_feature_rows":len(source_rows),"previous_clean_rows":len(existing_rows),"reused_exact_rows":len(selected),"pending_reextraction":len(missing),
        "identical_duplicate_ids_collapsed":identical,"conflicting_duplicate_ids_left_pending":conflicting,
        "outside_lineage_ids_excluded":outside,"non_ok_wanted_rows_not_reused":sorted(set(non_ok)),"missing_or_conflicting_ids":missing,
        "source_cache_modified":False,"policy":"never choose a conflicting duplicate; official extractor rebuilds absent IDs"}
    (a.output/"sanitization_audit.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    print(f"FUTURE_COMPACT_SANITIZED reused={len(selected)} pending={len(missing)} identical_duplicates={len(identical)} conflicting={len(conflicting)} outside={len(outside)}")
if __name__=="__main__":main()
