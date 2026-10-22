#!/usr/bin/env python3
from pathlib import Path
from haig_model import read_csv
ROOT=Path("artifacts/care-p31rf-haig-source-v1");BASE=Path("artifacts/care-p31rf-c43-p0-complete-diagnostics-v1")
def main():
    if not (ROOT/"source_selection.json").exists():raise RuntimeError("BLOCKED_BY_CONTRACT: source selection not frozen")
    paths=(BASE/"frozen_future/label_free_predictions.csv",BASE/"within_year/2018_2019/oof_predictions.csv",BASE/"within_year/2020_2021/oof_predictions.csv")
    missing=[str(p) for p in paths if not p.exists() or not p.stat().st_size]
    if missing:raise RuntimeError("BLOCKED_BY_CONTRACT: missing future diagnostics:\n- "+"\n- ".join(missing))
    if len(read_csv(paths[0]))!=1989 or len(read_csv(paths[1]))+len(read_csv(paths[2]))!=1989:raise RuntimeError("BLOCKED_BY_CONTRACT: future cohort size changed")
    print("CARE_HAIG_FUTURE_PREFLIGHT_PASS rows=1989 source_selection_frozen=true")
if __name__=="__main__":main()
