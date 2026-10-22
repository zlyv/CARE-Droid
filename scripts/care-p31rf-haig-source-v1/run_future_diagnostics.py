#!/usr/bin/env python3
import json
from pathlib import Path
import joblib,numpy as np
from haig_model import *
ROOT=Path("artifacts/care-p31rf-haig-source-v1");BASE=Path("artifacts/care-p31rf-c43-p0-complete-diagnostics-v1")
def main():
    rows=read_csv(ROOT/"lineage/future_2018_2021.csv");graphs,fail=load_graphs(rows,ROOT/"graphs/future");bundle=joblib.load(ROOT/"models/frozen_source_bundle.joblib");pred,norms=predict_bundle(bundle,graphs)
    stage=read_csv(BASE/"frozen_future/label_free_predictions.csv");by={norm(r["sha256"]):r for r in stage};out=[]
    for i,r in enumerate(rows):
        sha=norm(r["sha256"]);s=by[sha];x={"sha256":sha,"year":int(float(s["year"])),"p_stage1":float(s["p_stage1"]),"route_DI":int(float(s["route_DI"]))}
        for a in ARMS:x[a+"__prob"]=float(pred[a][i]);x[a+"__embedding_norm"]=float(norms[a][i])
        out.append(x)
    write_csv(ROOT/"future/frozen_label_free_predictions.csv",out);write_csv(ROOT/"future/extraction_failures.csv",fail);(ROOT/"future/protocol.json").write_text(json.dumps({"protocol":"frozen 2017 to 2018-2021","source_selected_before_future_labels":json.loads((ROOT/"source_selection.json").read_text()),"future_labels_loaded_by_model":False,"all_arms_are_locked_posthoc_diagnostics":True},indent=2)+"\n")
    print("HAIG_FROZEN_FUTURE_LABEL_FREE_DONE rows=1989 labels_loaded=false")
if __name__=="__main__":main()

