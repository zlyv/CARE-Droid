#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path

ROOT = Path("artifacts/care-di25-graphglobal-query-future-v1")
RESOLVED = Path("artifacts/care-route-stage2-future-matrix-v1/preflight/resolved_inputs.json")
FUTURE_INPUT = Path("artifacts/care-route-stage2-future-matrix-v1/preflight/future_router_inputs_label_free.csv")
SOURCE_ROOT = Path("artifacts/care-di25-graphglobal-path-source-v1")
YEARS = (2018, 2019, 2020, 2021)
YEAR_COUNTS = {2018: 793, 2019: 399, 2020: 397, 2021: 400}
ROUTE_COUNTS = {2018: 198, 2019: 100, 2020: 99, 2021: 100}

def norm(x): return str(x or "").strip().lower()
def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as h: return list(csv.DictReader(h))
def write_csv(path, rows):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(rows)
def sha256(path):
    d=hashlib.sha256()
    with Path(path).open("rb") as h:
        for b in iter(lambda:h.read(1024*1024),b""): d.update(b)
    return d.hexdigest()
def unique(rows, expected):
    out={}
    for row in rows:
        key=norm(row.get("sha256") or row.get("sample_id"))
        if not key or key in out: raise RuntimeError("missing/duplicate SHA "+key)
        out[key]=row
    if len(rows)!=expected or len(out)!=expected: raise RuntimeError(f"cohort changed rows={len(rows)} unique={len(out)}")
    return out
