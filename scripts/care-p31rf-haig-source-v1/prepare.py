#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
ROOT=Path("artifacts/care-p31rf-haig-source-v1")
SRC_REF=Path("artifacts/care-p31-stage1-rf-selection-source-v1/preflight/C_RF_RFROUTE/source_predictions.csv")
SRC_CAND=(Path("artifacts/care-source-local-assets-v2/source_lineage_2017.csv"),Path("artifacts/care-source-recovery-stage2only-v1/source_lineage_2017.csv"),Path("artifacts/clean-local-context-2017/lineage_2017.csv"))
FUT_CAND=(Path("artifacts/care-di25-noego-safe-authority-full-v1/preflight/future_lineage_2018_2021_exact.csv"),Path("artifacts/care-route-stage2-future-matrix-v1/preflight/future_lineage_2018_2021_exact.csv"),Path("artifacts/care-future-local-assets-v2/future_lineage_2018_2021.csv"))
LABEL=Path("artifacts/care-litelocal-p31-diagnostics-v1/within_year/oof_predictions.csv")
def rows(p):
    with p.open(encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def find(candidates,name):
    p=next((x for x in candidates if x.exists() and x.stat().st_size),None)
    if p is None:raise RuntimeError("BLOCKED_BY_CONTRACT: missing "+name+":\n- "+"\n- ".join(map(str,candidates)))
    return p
def write(p,data):
    fields=[]
    for r in data:
        for k in r:
            if k not in fields:fields.append(k)
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf-8",newline="") as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(data)
def aligned(ids,data):
    by={str(r.get("sha256","")).casefold():r for r in data};missing=[x for x in ids if x not in by]
    if missing:raise RuntimeError(f"BLOCKED_BY_CONTRACT: lineage lacks {len(missing)} SHA rows")
    out=[by[x] for x in ids]
    for r in out:
        if not r.get("absolute_path"):raise RuntimeError("BLOCKED_BY_CONTRACT: lineage absolute_path empty")
    return out
def main():
    p=argparse.ArgumentParser();p.add_argument("--scope",choices=("source","future","all"),required=True);a=p.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
    resolved={}
    old=ROOT/"resolved_inputs.json"
    if old.exists():resolved=json.loads(old.read_text())
    if a.scope in ("source","all"):
        src_ids=[r["sha256"].casefold() for r in rows(SRC_REF)]
        if len(src_ids)!=799:raise RuntimeError("BLOCKED_BY_CONTRACT: source cohort changed")
        src=find(SRC_CAND,"source lineage");write(ROOT/"lineage/source_2017.csv",aligned(src_ids,rows(src)));resolved.update({"source_lineage":str(src.resolve()),"source_rows":799})
        print("HAIG_SOURCE_PREPARE_PASS source_rows=799 future_rows_loaded=0")
    if a.scope in ("future","all"):
        # This branch must be invoked only after source_selection.json is frozen.
        if not (ROOT/"source_selection.json").exists():raise RuntimeError("BLOCKED_BY_CONTRACT: source selection is not frozen")
        labels=rows(LABEL);fut_ids=[r["sha256"].casefold() for r in labels]
        if len(fut_ids)!=1989:raise RuntimeError("BLOCKED_BY_CONTRACT: future cohort changed")
        fut=find(FUT_CAND,"future lineage");write(ROOT/"lineage/future_2018_2021.csv",aligned(fut_ids,rows(fut)));resolved.update({"future_lineage":str(fut.resolve()),"future_rows":1989,"future_year_counts":{"2018":793,"2019":399,"2020":397,"2021":400}})
        print("HAIG_FUTURE_PREPARE_PASS future_rows=1989 source_selection_frozen=true")
    (ROOT/"resolved_inputs.json").write_text(json.dumps(resolved,indent=2)+"\n")
if __name__=="__main__":main()

