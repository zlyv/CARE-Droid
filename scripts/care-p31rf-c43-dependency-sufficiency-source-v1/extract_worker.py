#!/usr/bin/env python3
import argparse, csv, subprocess, sys
from pathlib import Path

ROOT=Path("artifacts/care-p31rf-c43-dependency-sufficiency-source-v1")
RAW=ROOT/"dependency_raw"
HERE=Path(__file__).resolve().parent
EXTRACTOR=HERE/"vendor/scripts/p70_build_verified_dependency_ag.py"
MAPPING=HERE/"vendor/configs/protocol/semantic_mapping_v1.yaml"

def row_count(path):
    with path.open(encoding="utf-8-sig",newline="") as h:return sum(1 for _ in csv.DictReader(h))

def main():
    p=argparse.ArgumentParser();p.add_argument("--worker",type=int,required=True);p.add_argument("--workers",type=int,default=2);args=p.parse_args()
    batches=sorted(x for x in RAW.glob("batch_*") if x.is_dir())
    mine=[x for i,x in enumerate(batches) if i%args.workers==args.worker]
    print(f"dependency worker {args.worker}/{args.workers} batches={len(mine)}",flush=True)
    for number,directory in enumerate(mine,1):
        lineage=directory/"lineage.csv";units=directory/"result_centered_behavior_units.jsonl";summary=directory/"result_centered_slice_summary.csv";marker=directory/"COMPLETE"
        expected=row_count(lineage)
        if marker.exists() and units.exists() and summary.exists() and row_count(summary)==expected:
            print(f"[resume] {directory.name} complete {number}/{len(mine)}",flush=True);continue
        command=[sys.executable,str(EXTRACTOR),"--lineage",str(lineage),"--mapping",str(MAPPING),"--output",str(directory),"--limit",str(expected),"--progress-every",str(expected),"--min-free-gb","2"]
        result=subprocess.run(command,check=False)
        if result.returncode!=0:raise RuntimeError(f"extractor failed for {directory.name}: {result.returncode}")
        if not units.exists() or not summary.exists() or row_count(summary)!=expected:raise RuntimeError(f"incomplete extractor output for {directory.name}")
        marker.write_text("complete\n",encoding="utf-8")
        print(f"[done] {directory.name} {number}/{len(mine)}",flush=True)
    print(f"P31RF_DEPENDENCY_WORKER_DONE {args.worker}",flush=True)

if __name__=="__main__":main()

