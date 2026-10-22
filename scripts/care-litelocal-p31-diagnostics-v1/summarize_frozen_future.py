#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score,f1_score,roc_auc_score
from p31_core import label

OUT=Path("artifacts/care-litelocal-p31-diagnostics-v1")
def read(path):
    with Path(path).open("r",encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def norm(v):return str(v or "").strip().casefold()
def metrics(y,p):
    pred=p>=.5;return float(balanced_accuracy_score(y,pred)),float(roc_auc_score(y,p)),float(f1_score(y,pred,zero_division=0)),int(np.sum(pred!=y))
def main():
    cfg=json.loads((OUT/"preflight/resolved_inputs.json").read_text());pred=read(OUT/"frozen_future/label_free_predictions.csv");lab=read(cfg["future_label_source_posthoc_only"]);lm={norm(r.get("sha256") or r.get("sample_id")):label(r.get("metadata_label") or r.get("label")) for r in lab}
    ids=np.asarray([norm(r["sha256"]) for r in pred]);years=np.asarray([int(r["year"]) for r in pred]);p=np.asarray([float(r["p_stage1"]) for r in pred]);route=np.asarray([int(r["route_DI"])==1 for r in pred]);y=np.asarray([lm[s] for s in ids]);rows=[]
    for scope,idx in [(str(v),years==v) for v in sorted(set(years))]+[("pooled",np.ones(len(y),bool))]:
        ba,auc,f1,errors=metrics(y[idx],p[idx]);caught=int(np.sum(((p>=.5)!=y)&route&idx));total=int(np.sum(((p>=.5)!=y)&idx));rows.append({"scope":scope,"n":int(idx.sum()),"routed":int(np.sum(route&idx)),"BA":ba,"AUC":auc,"F1":f1,"errors":errors,"captured_errors":caught,"error_recall_at_25":caught/max(total,1),"routing_precision":caught/max(int(np.sum(route&idx)),1)})
    path=OUT/"frozen_future/metrics.csv"
    with path.open("w",encoding="utf-8",newline="") as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=["CARE-Droid P31 Mean-Committee frozen 2017-to-future diagnostic","The frozen-future model and route never loaded future labels; label values were joined to its saved predictions only by this summarizer. The separate same-year protocol uses annual training labels by design."]+[f"{r['scope']}: BA={r['BA']:.6f} AUC={r['AUC']:.6f} F1={r['F1']:.6f} errors={r['errors']} routed={r['routed']} captured={r['captured_errors']} recall25={r['error_recall_at_25']:.6f}" for r in rows]
    (OUT/"frozen_future/REPORT.txt").write_text("\n".join(lines)+"\n");print("LITELOCAL_P31_FROZEN_FUTURE_SUMMARY_PASS");print("\n".join(lines))
if __name__=="__main__":main()
