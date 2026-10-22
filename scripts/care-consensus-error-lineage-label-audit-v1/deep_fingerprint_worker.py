#!/usr/bin/env python3
from __future__ import annotations
import argparse,gzip,json,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument("--apk",required=True);p.add_argument("--sha256",required=True);p.add_argument("--output",required=True);p.add_argument("--verified-dir",required=True)
    a=p.parse_args();verified=Path(a.verified_dir).resolve();sys.path.insert(0,str(verified))
    # This imports the already-validated source-lineage implementation rather than forking its algorithm.
    from fingerprints import fingerprint_apk
    record=fingerprint_apk(Path(a.apk).resolve(),a.sha256);output=Path(a.output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    temp=output.with_suffix(output.suffix+".tmp")
    with gzip.open(temp,"wt",encoding="utf-8") as h:json.dump(record,h,sort_keys=True,separators=(",",":"))
    temp.replace(output)
if __name__=="__main__":main()
