#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from pathlib import Path

OUT=Path("artifacts/care-litesemantic-noshortcut-source-v1")
LINEAGES=(Path("artifacts/care-source-local-assets-v2/source_lineage_2017.csv"),Path("artifacts/care-source-recovery-stage2only-v1/source_lineage_2017.csv"),Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/source_lineage_2017.csv"))
MAPPINGS=(Path("scripts/care-source-recovery-stage2only-v1/extractor_repo/configs/protocol/semantic_mapping_v1.yaml"),Path("scripts/care-route-stage2-future-matrix-v1/configs/semantic_mapping_v1.yaml"))
ROUTER=Path("artifacts/care-source-di25-reconstructed-v1/source_router_inputs.csv")

def read(p):
    with p.open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def norm(v):return str(v or "").strip().lower()
def first(paths):
    for p in paths:
        if p.is_file() and p.stat().st_size>10:return p
    return None

def main():
    lineage=first(LINEAGES);mapping=first(MAPPINGS)
    if lineage is None:raise RuntimeError("BLOCKED_BY_CONTRACT: no canonical 2017 source lineage found")
    if mapping is None:raise RuntimeError("BLOCKED_BY_CONTRACT: semantic_mapping_v1.yaml not found")
    if not ROUTER.is_file():raise RuntimeError("BLOCKED_BY_CONTRACT: frozen source router missing")
    lr=read(lineage);rr=read(ROUTER);lids=[norm(r.get("sha256")) for r in lr];rids=[norm(r.get("sha256")) for r in rr]
    paths=[Path(r.get("absolute_path") or r.get("apk_path") or "") for r in lr]
    years={str(r.get("metadata_year") or r.get("year") or "").strip() for r in lr}
    folds=[int(float(r["fold"])) for r in rr];routes=[int(float(r["route_DI"])) for r in rr]
    if len(lr)!=799 or len(set(lids))!=799 or len(rr)!=799 or len(set(rids))!=799 or set(lids)!=set(rids) or years!={"2017"}:raise RuntimeError("BLOCKED_BY_CONTRACT: source lineage/router mismatch")
    missing_paths=[str(p) for p in paths if not p.is_file()]
    if missing_paths:raise RuntimeError("BLOCKED_BY_CONTRACT: source APK paths missing; first="+missing_paths[0])
    if {f:folds.count(f) for f in set(folds)}!={1:160,2:160,3:160,4:160,5:159} or sum(routes)!=200:raise RuntimeError("BLOCKED_BY_CONTRACT: fold/DI25 contract changed")
    OUT.mkdir(parents=True,exist_ok=True)
    resolved={"passed":True,"lineage":str(lineage.resolve()),"mapping":str(mapping.resolve()),"router":str(ROUTER.resolve()),"source_rows":799,"year":2017,"apk_paths_exist":True,"route_DI":200,"future_rows_loaded":0,"callgraph_constructed":False,"identity_features":False,"raw_size_features_in_model":False}
    (OUT/"resolved_inputs.json").write_text(json.dumps(resolved,indent=2)+"\n",encoding="utf-8")
    print("LITESEMANTIC_NOSHORTCUT_PREFLIGHT_PASS")
    print("source_rows=799 year=2017 route_DI=200 apk_paths_exist=true")
    print("lineage="+str(lineage.resolve()));print("mapping="+str(mapping.resolve()))
    print("future_rows_loaded=0 callgraph_constructed=false identity_features=false raw_size_features_in_model=false")

if __name__=="__main__":main()
