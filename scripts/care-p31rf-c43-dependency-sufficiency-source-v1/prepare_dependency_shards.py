#!/usr/bin/env python3
import csv, json
from pathlib import Path

REF=Path("artifacts/care-p31-stage1-rf-selection-source-v1/preflight/C_RF_RFROUTE/source_predictions.csv")
LINEAGE_CANDIDATES=(
    Path("artifacts/care-source-local-assets-v2/source_lineage_2017.csv"),
    Path("artifacts/care-source-recovery-stage2only-v1/source_lineage_2017.csv"),
    Path("artifacts/clean-local-context-2017/lineage_2017.csv"),
)
ROOT=Path("artifacts/care-p31rf-c43-dependency-sufficiency-source-v1")
RAW=ROOT/"dependency_raw"
BATCH=25

def rows(path):
    with path.open(encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))

def write(path,data,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(data)

def main():
    lineage=next((p for p in LINEAGE_CANDIDATES if p.exists() and p.stat().st_size),None)
    if lineage is None:raise RuntimeError("BLOCKED_BY_CONTRACT: no 2017 source lineage with APK paths")
    wanted=[str(r["sha256"]).casefold() for r in rows(REF)]
    source=rows(lineage);by_sha={str(r.get("sha256","")).casefold():r for r in source}
    missing=[sha for sha in wanted if sha not in by_sha]
    if missing:raise RuntimeError(f"BLOCKED_BY_CONTRACT: lineage lacks {len(missing)} source SHA rows")
    selected=[by_sha[sha] for sha in wanted]
    if any(not str(r.get("absolute_path","")).strip() for r in selected):raise RuntimeError("BLOCKED_BY_CONTRACT: empty APK absolute_path")
    fields=list(selected[0])
    for start in range(0,len(selected),BATCH):
        index=start//BATCH;directory=RAW/f"batch_{index:03d}"
        write(directory/"lineage.csv",selected[start:start+BATCH],fields)
    manifest={"source_rows":len(selected),"batch_size":BATCH,"batch_count":(len(selected)+BATCH-1)//BATCH,"lineage":str(lineage),"selection":"fixed P31-RF source SHA order; labels not used"}
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/"dependency_extraction_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print("P31RF_DEPENDENCY_EXTRACTION_SHARDS_PASS")
    print(f"source_rows={len(selected)} batches={manifest['batch_count']} batch_size={BATCH}")

if __name__=="__main__":main()

