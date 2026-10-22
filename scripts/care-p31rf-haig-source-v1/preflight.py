#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
from haig_model import read_csv
ROOT=Path("artifacts/care-p31rf-haig-source-v1")
REQUIRED=(
 "artifacts/care-p31-stage1-rf-selection-source-v1/preflight/C_RF_RFROUTE/source_predictions.csv",
)
def main():
    missing=[p for p in REQUIRED if not Path(p).exists() or not Path(p).stat().st_size]
    if missing:raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty inputs:\n- "+"\n- ".join(missing))
    src=read_csv(REQUIRED[0])
    if len(src)!=799:raise RuntimeError("BLOCKED_BY_CONTRACT: source cohort size changed")
    y=np.asarray([int(float(r["label"])) for r in src]);p=np.asarray([float(r["Global24__prob"]) for r in src]);route=np.asarray([int(float(r["A25__routed"]))==1 for r in src])
    if route.sum()!=200 or np.sum((p>=.5)!=y)!=58 or np.sum(route&((p>=.5)!=y))!=45:raise RuntimeError("BLOCKED_BY_CONTRACT: source P31-RF/DI25 contract changed")
    protocol={"stage1":"P31 calibrated RF","router":"RF-anchored MeanOpposition DI25 Top25%","source_rows":799,"source_folds":"160,160,160,160,159","future_rows":1989,"source_selection_year":2017,"future_years":[2018,2019,2020,2021],"future_used_for_selection":False,"same_host_semantics":"ordered local interaction, not data flow","host_call_semantics":"direct resolved Host invocation, not data flow","max_delta_unchanged":2.25,"P70_P71_preserved":True}
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    print("CARE_HAIG_PREFLIGHT_PASS source_rows=799 future_rows_loaded=0")
if __name__=="__main__":main()
