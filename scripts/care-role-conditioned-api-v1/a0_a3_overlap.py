#!/usr/bin/env python3
import csv
from pathlib import Path
import numpy as np
from role_api_core import write_csv,norm
ROOT=Path("artifacts/care-role-conditioned-api-v1");OLD=Path("artifacts/care-p31rf-c43-haig-host-ablation-source-v1")
def read(p):
    with Path(p).open(encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def prob(rows,tag):
    col=next(x for x in rows[0] if x.endswith("__prob") and tag in x);return {norm(r["sha256"]):float(r[col]) for r in rows}
def main():
    ref=read("artifacts/care-p31-stage1-rf-selection-source-v1/preflight/C_RF_RFROUTE/source_predictions.csv");a0=prob(read(OLD/"arms/A0_ROLE_ONLY/source_predictions.csv"),"A0_ROLE_ONLY");a3=prob(read(OLD/"arms/A3_ROLE_PLUS_H3/source_predictions.csv"),"A3_ROLE_PLUS_H3");out=[]
    for r in ref:
        s=norm(r["sha256"]);y=int(float(r["label"]));base=float(r["Global24__prob"]);b=(base>=.5)==y;x=(a0[s]>=.5)==y;z=(a3[s]>=.5)==y
        out.append({"sha256":s,"fold":r["fold"],"label":y,"base_correct":int(b),"A0_correct":int(x),"A3_correct":int(z),"both_fixed":int(not b and x and z),"A0_only_fixed":int(not b and x and not z),"A3_only_fixed":int(not b and z and not x),"both_broken":int(b and not x and not z),"A0_only_broken":int(b and not x and z),"A3_only_broken":int(b and x and not z)})
    write_csv(ROOT/"a0_a3_sample_overlap.csv",out);print("A0_A3_OVERLAP_DONE",{k:sum(int(r[k]) for r in out) for k in out[0] if k.endswith("fixed") or k.endswith("broken")})
if __name__=="__main__":main()
