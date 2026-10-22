#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
from haig_model import *
ROOT=Path("artifacts/care-p31rf-haig-source-v1");BASE=Path("artifacts/care-p31rf-c43-p0-complete-diagnostics-v1")
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--years",default="2018,2019,2020,2021");ap.add_argument("--output",type=Path,default=ROOT/"future/within_year_all");args=ap.parse_args();requested=tuple(int(v) for v in args.years.split(","))
    rows=read_csv(ROOT/"lineage/future_2018_2021.csv");graphs,_=load_graphs(rows,ROOT/"graphs/future");contracts=[]
    for g in ("2018_2019","2020_2021"):contracts+=read_csv(BASE/f"within_year/{g}/oof_predictions.csv")
    by={norm(r["sha256"]):r for r in contracts};c=[by[norm(r["sha256"])] for r in rows];y=np.array([int(float(r["label"])) for r in c]);years=np.array([int(float(r["year"])) for r in c]);folds=np.array([int(float(r["fold"])) for r in c]);base=np.array([float(r["p_stage1"]) for r in c]);route=np.array([int(float(r["route_DI"]))==1 for r in c]);pred={a:np.full(len(rows),np.nan) for a in ARMS};foldout=[]
    selected=np.isin(years,requested)
    for year in requested:
        for f in range(1,6):
            tr=np.flatnonzero((years==year)&(folds!=f));va=np.flatnonzero((years==year)&(folds==f));bundle=fit_bundle([graphs[i] for i in tr],y[tr],year*100+f);p,_=predict_bundle(bundle,[graphs[i] for i in va])
            for a in ARMS:pred[a][va]=p[a];foldout.append({"year":year,"fold":f,"arm":a,"n":len(va),**metric(y[va],p[a]),**direction(y[va],base[va],route[va],p[a])})
            print(f"HAIG same-year {year} fold {f}/5 done",flush=True)
    out=[]
    for i,r in enumerate(rows):
        if not selected[i]:continue
        x={"sha256":norm(r["sha256"]),"year":int(years[i]),"fold":int(folds[i]),"label":int(y[i]),"p_stage1":float(base[i]),"route_DI":int(route[i])}
        for a in ARMS:x[a+"__prob"]=float(pred[a][i])
        out.append(x)
    write_csv(args.output/"oof_predictions.csv",out);write_csv(args.output/"fold_results.csv",foldout);print(f"HAIG_WITHIN_YEAR_GROUP_DONE years={args.years} rows={len(out)}")
if __name__=="__main__":main()

