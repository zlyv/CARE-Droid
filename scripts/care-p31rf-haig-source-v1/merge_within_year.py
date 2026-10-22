#!/usr/bin/env python3
from pathlib import Path
from haig_model import read_csv,write_csv,norm
ROOT=Path("artifacts/care-p31rf-haig-source-v1/future")
def main():
    pred=[];fold=[]
    for g in ("within_2018_2019","within_2020_2021"):
        pred+=read_csv(ROOT/g/"oof_predictions.csv");fold+=read_csv(ROOT/g/"fold_results.csv")
    if len(pred)!=1989 or len({norm(r["sha256"]) for r in pred})!=1989:raise RuntimeError("BLOCKED_BY_CONTRACT: within-year merge mismatch")
    write_csv(ROOT/"within_year_oof_predictions.csv",pred);write_csv(ROOT/"within_year_fold_results.csv",fold);print("HAIG_WITHIN_YEAR_DONE rows=1989")
if __name__=="__main__":main()
