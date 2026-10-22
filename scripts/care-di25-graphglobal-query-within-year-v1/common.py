#!/usr/bin/env python3
from __future__ import annotations
import csv, math
from pathlib import Path

ROOT=Path("artifacts/care-di25-graphglobal-query-within-year-v1")
MATRIX_ROOT=Path("artifacts/care-route-stage2-within-year-matrix-v1")
ROUTER=MATRIX_ROOT/"shared/within_year_router_inputs.csv"
RESOLVED=MATRIX_ROOT/"preflight/resolved_inputs.json"
YEARS=(2018,2019,2020,2021)
YEAR_COUNTS={2018:793,2019:399,2020:397,2021:400}
ROUTE_COUNTS={2018:198,2019:100,2020:99,2021:100}
ARMS=("N0_NO_GRAPHGLOBAL","Q_QUERY_ONLY")

def norm(v):return str(v or "").strip().lower()
def read_csv(path):
    with Path(path).open("r",encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields:fields.append(key)
    with path.open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
def unique(rows,expected=1989):
    out={}
    for row in rows:
        key=norm(row.get("sha256") or row.get("sample_id"))
        if not key or key in out:raise RuntimeError("missing/duplicate SHA "+key)
        out[key]=row
    if len(rows)!=expected or len(out)!=expected:raise RuntimeError(f"cohort changed rows={len(rows)} unique={len(out)}")
    return out
def exact_p(a,b):
    n=a+b
    if n==0:return 1.0
    k=min(a,b)
    return min(1.0,2.0*sum(math.comb(n,i) for i in range(k+1))/(2**n))
