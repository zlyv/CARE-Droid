#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np

ROOT=Path("artifacts/care-d0-correction-score-calibration-v1")
BASE=Path("artifacts/care-p31rf-c43-haig-host-ablation-source-v1/arms/A0_ROLE_ONLY/source_predictions.csv")
METHOD="P31RF__RFRoute__A0_ROLE_ONLY__Bounded"
MAX_DELTA=2.25
def read_csv(p):
    with Path(p).open(encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def write_csv(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:p.write_text("",encoding="utf-8");return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with p.open("w",encoding="utf-8",newline="") as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
def norm(x):return str(x).strip().casefold()
def logit(p):p=np.clip(np.asarray(p,float),1e-8,1-1e-8);return np.log(p/(1-p))

