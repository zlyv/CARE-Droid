#!/usr/bin/env python3
from __future__ import annotations
import csv, json
from pathlib import Path

OUT = Path("artifacts/care-litelocal-semantic-source-v1")
LINEAGES = (
    Path("artifacts/care-source-local-assets-v2/source_lineage_2017.csv"),
    Path("artifacts/care-source-recovery-stage2only-v1/source_lineage_2017.csv"),
    Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/source_lineage_2017.csv"),
)
MAPPINGS = (
    Path("scripts/care-source-recovery-stage2only-v1/extractor_repo/configs/protocol/semantic_mapping_v1.yaml"),
    Path("scripts/care-route-stage2-future-matrix-v1/configs/semantic_mapping_v1.yaml"),
)
ROUTER = Path("artifacts/care-source-di25-reconstructed-v1/source_router_inputs.csv")

def read(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))
def norm(value): return str(value or "").strip().casefold()
def first(paths):
    return next((p for p in paths if p.is_file() and p.stat().st_size > 10), None)

def main():
    lineage, mapping = first(LINEAGES), first(MAPPINGS)
    if lineage is None or mapping is None or not ROUTER.is_file():
        raise RuntimeError("BLOCKED_BY_CONTRACT: canonical 2017 lineage/mapping/router missing")
    source, router = read(lineage), read(ROUTER)
    source_ids = [norm(r.get("sha256")) for r in source]
    router_ids = [norm(r.get("sha256")) for r in router]
    years = {str(r.get("metadata_year") or r.get("year") or "").strip() for r in source}
    folds = [int(float(r["fold"])) for r in router]
    route = [int(float(r["route_DI"])) for r in router]
    paths = [Path(r.get("absolute_path") or r.get("apk_path") or "") for r in source]
    expected_folds = {1:160, 2:160, 3:160, 4:160, 5:159}
    route_per_fold = {f:sum(int(float(r["route_DI"])) for r in router if int(float(r["fold"]))==f) for f in expected_folds}
    if len(source) != 799 or len(set(source_ids)) != 799 or len(router) != 799 or set(source_ids) != set(router_ids) or years != {"2017"}:
        raise RuntimeError("BLOCKED_BY_CONTRACT: source lineage/router mismatch")
    if {f:folds.count(f) for f in sorted(set(folds))} != expected_folds or sum(route) != 200 or route_per_fold != {1:40,2:40,3:40,4:40,5:40}:
        raise RuntimeError("BLOCKED_BY_CONTRACT: fold/DI25 contract changed")
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise RuntimeError("BLOCKED_BY_CONTRACT: source APK missing: " + missing[0])
    OUT.mkdir(parents=True, exist_ok=True)
    resolved = {"passed":True, "lineage":str(lineage.resolve()), "mapping":str(mapping.resolve()),
                "router":str(ROUTER.resolve()), "source_rows":799, "year":2017,
                "future_rows_loaded":0, "callgraph_constructed":False,
                "analysis_object_constructed":False, "stage2_training_invoked":False}
    (OUT/"resolved_inputs.json").write_text(json.dumps(resolved, indent=2)+"\n", encoding="utf-8")
    print("LITELOCAL_SEMANTIC_SOURCE_PREFLIGHT_PASS")
    print("source_rows=799 year=2017 route_DI=200")
    print("fold_sizes=1:160,2:160,3:160,4:160,5:159")
    print("route_DI_per_fold=1:40,2:40,3:40,4:40,5:40")
    print("future_rows_loaded=0 callgraph_constructed=false analysis_object_constructed=false")

if __name__ == "__main__": main()

