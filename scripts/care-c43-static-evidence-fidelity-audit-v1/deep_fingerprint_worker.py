#!/usr/bin/env python3
from __future__ import annotations
import argparse,gzip,json,sys
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument("--apk",required=True);p.add_argument("--sha256",required=True);p.add_argument("--output",required=True);p.add_argument("--verified-dir",required=True)
    a=p.parse_args();sys.path.insert(0,str(Path(a.verified_dir).resolve()))
    from fingerprints import fingerprint_apk
    record=fingerprint_apk(Path(a.apk).resolve(),a.sha256);out=Path(a.output).resolve();out.parent.mkdir(parents=True,exist_ok=True)
    temp=out.with_suffix(out.suffix+".tmp")
    with gzip.open(temp,"wt",encoding="utf-8") as h:json.dump(record,h,sort_keys=True,separators=(",",":"))
    temp.replace(out)
if __name__=="__main__":main()
