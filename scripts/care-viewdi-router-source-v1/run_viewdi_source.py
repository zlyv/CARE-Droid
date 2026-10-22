#!/usr/bin/env python3
"""Strict 2017 OOF leave-one-view-out decision-instability router audit."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
for path in (
    HERE,
    SCRIPTS/"care-multiview-conformal-router-source-v1",
    SCRIPTS/"care-rolegraph-conditioned-v4-script",
    SCRIPTS/"care-rgcondattn-dedup-bounded-source-v1",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import run_multiview_conformal_router_source_v1 as mv
except Exception as exc:
    raise SystemExit("BLOCKED_BY_CONTRACT: missing established 2017 Global24 loader: "+str(exc))

from viewdi_core import (MEMBERS, clipped_logit, decision_instability,
                         exact_binomial_two_sided, feature_groups,
                         groupwise_top_fraction, route_metrics)

EXPECTED_FOLDS = {1:160, 2:160, 3:160, 4:160, 5:159}
EXPECTED_ROUTE = {1:40, 2:40, 3:40, 4:40, 5:40}


def norm(value) -> str:
    return str(value or "").strip().lower()


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise RuntimeError("BLOCKED_BY_CONTRACT: missing "+str(path))
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def make_lr(seed: int):
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=1.0, penalty="l2", solver="liblinear",
                                     max_iter=5000, random_state=int(seed))),
    ])


def calibrated_lovo(x_train, y_train, x_validation, keep_index,
                    seed: int, inner_folds: int):
    split = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    raw_oof = np.full(len(y_train), np.nan, dtype=np.float64)
    for inner, (fit, cal) in enumerate(split.split(x_train, y_train), start=1):
        model = make_lr(seed+100*inner)
        model.fit(x_train[fit][:, keep_index], y_train[fit])
        raw_oof[cal] = model.predict_proba(x_train[cal][:, keep_index])[:, 1]
    if not np.isfinite(raw_oof).all():
        raise RuntimeError("incomplete inner OOF LOVO probabilities")
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000,
                                    random_state=seed+999)
    calibrator.fit(clipped_logit(raw_oof).reshape(-1, 1), y_train)
    final = make_lr(seed+777)
    final.fit(x_train[:, keep_index], y_train)
    raw_target = final.predict_proba(x_validation[:, keep_index])[:, 1]
    calibrated = calibrator.predict_proba(clipped_logit(raw_target).reshape(-1, 1))[:, 1]
    return np.clip(calibrated, 1e-8, 1.0-1e-8), {
        "inner_oof_n": int(len(raw_oof)),
        "retained_dimension": int(len(keep_index)),
        "raw_oof_mean": float(raw_oof.mean()),
        "target_raw_mean": float(raw_target.mean()),
        "target_calibrated_mean": float(calibrated.mean()),
    }


def validate_router(rows: list[dict]):
    required = {"sha256","year","label","fold","p_stage1","route_DI"}
    if len(rows) != 799 or not rows or not required.issubset(rows[0]):
        raise RuntimeError("BLOCKED_BY_CONTRACT: canonical source router contract failed")
    ids = [norm(r["sha256"]) for r in rows]
    years = {int(float(r["year"])) for r in rows}
    folds = np.asarray([int(float(r["fold"])) for r in rows])
    route = np.asarray([int(float(r["route_DI"])) for r in rows])
    sizes = {f:int(np.sum(folds==f)) for f in sorted(set(folds))}
    routed = {f:int(np.sum(route[folds==f])) for f in sorted(set(folds))}
    if len(set(ids)) != 799 or years != {2017} or sizes != EXPECTED_FOLDS or routed != EXPECTED_ROUTE:
        raise RuntimeError("BLOCKED_BY_CONTRACT: frozen DI25 cohort/route changed: "+repr((years,sizes,routed)))
    return {sha:row for sha,row in zip(ids,rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("artifacts/p1-features/features.csv"))
    parser.add_argument("--schema", type=Path, default=Path("artifacts/p1-features/feature_schema.json"))
    parser.add_argument("--p72-oof", type=Path, default=Path("artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv"))
    parser.add_argument("--router", type=Path, default=Path("artifacts/care-source-di25-reconstructed-v1/source_router_inputs.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/care-viewdi-router-source-v1"))
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--budget", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # Established loader reads only the fixed 2017 source assets here.
    manifest, source_rows, _, compact_by_sha, compact_schema, original_names = mv.v4.audit.load_inputs(args)
    if len(manifest) != 799 or {str(r.get("metadata_year","")).strip() for r in source_rows} != {"2017"}:
        raise RuntimeError("BLOCKED_BY_CONTRACT: loader returned a non-2017 cohort")
    router = validate_router(read_csv(args.router))
    ids = [norm(r["sha256"]) for r in manifest]
    if set(ids) != set(router):
        raise RuntimeError("BLOCKED_BY_CONTRACT: loader/router SHA mismatch")
    y = np.asarray([int(float(router[s]["label"])) for s in ids], dtype=np.int64)
    folds = np.asarray([int(float(router[s]["fold"])) for s in ids], dtype=np.int64)
    p_stage1 = np.asarray([float(router[s]["p_stage1"]) for s in ids], dtype=np.float64)
    frozen_di = np.asarray([int(float(router[s]["route_DI"]))==1 for s in ids], dtype=bool)
    for i,row in enumerate(manifest):
        if int(row["label"]) != y[i] or int(row["fold"]) != folds[i]:
            raise RuntimeError("BLOCKED_BY_CONTRACT: label/fold mismatch for "+ids[i])

    probabilities = np.full((799,4), np.nan, dtype=np.float64)
    audit=[]; canonical_names=None
    for fold in sorted(np.unique(folds)):
        train=np.flatnonzero(folds!=fold); val=np.flatnonzero(folds==fold)
        x_train,x_val,names=mv.global_matrices(
            [source_rows[i] for i in train], [source_rows[i] for i in val],
            compact_by_sha, compact_schema, original_names)
        if canonical_names is None:
            canonical_names=list(names)
        elif list(names)!=canonical_names:
            raise RuntimeError("Global24 names changed across outer folds")
        groups=feature_groups(list(names))
        for column,member in enumerate(MEMBERS):
            p,detail=calibrated_lovo(x_train,y[train],x_val,groups[member],
                                     args.seed+100000*int(fold)+10000*column,
                                     args.inner_folds)
            probabilities[val,column]=p
            audit.append({"outer_fold":int(fold),"member":member,
                          "outer_train_n":int(len(train)),"outer_validation_n":int(len(val)),
                          "outer_validation_labels_used":False,**detail})
        print(f"ViewDI outer fold {fold}/5 done",flush=True)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("incomplete ViewDI OOF probabilities")

    detail=decision_instability(p_stage1,probabilities)
    view_route=groupwise_top_fraction(detail["score"],folds,args.budget)
    a25_score=1.0-2.0*np.abs(p_stage1-0.5)
    a25_route=groupwise_top_fraction(a25_score,folds,args.budget)
    routes={"A25":a25_route,"Current_DI25":frozen_di,"LOVO_ViewDI25":view_route}
    scores={"A25":a25_score,"Current_DI25":np.asarray([float(router[s].get("U_DI",0.0)) for s in ids]),
            "LOVO_ViewDI25":detail["score"]}
    metrics=[route_metrics(name,y,p_stage1,routes[name],scores[name]) for name in routes]
    per_fold=[]
    for fold in sorted(np.unique(folds)):
        index=folds==fold
        for name in routes:
            row=route_metrics(name,y[index],p_stage1[index],routes[name][index],scores[name][index])
            per_fold.append({"fold":int(fold),**row})

    error=(p_stage1>=0.5)!=y
    di_only=error & frozen_di & ~view_route
    view_only=error & view_route & ~frozen_di
    both=error & view_route & frozen_di
    pvalue=exact_binomial_two_sided(int(di_only.sum()),int(view_only.sum()))
    paired={
        "comparison":"LOVO_ViewDI25 vs Current_DI25",
        "both_capture_error":int(both.sum()),
        "current_DI_only_error":int(di_only.sum()),
        "LOVO_ViewDI_only_error":int(view_only.sum()),
        "capture_delta":int(view_only.sum()-di_only.sum()),
        "exact_p_value":pvalue,
        "more_stage1_errors_captured":bool(view_only.sum()>di_only.sum()),
        "statistically_supported_at_0p05":bool(view_only.sum()>di_only.sum() and pvalue<0.05),
    }
    sample=[]
    for i,sha in enumerate(ids):
        row={"sha256":sha,"year":2017,"fold":int(folds[i]),"label":int(y[i]),
             "p_stage1":float(p_stage1[i]),"stage1_error":int(error[i]),
             "route_current_DI25":int(frozen_di[i]),"route_LOVO_ViewDI25":int(view_route[i]),
             "U_LOVO_ViewDI":float(detail["score"][i]),"LOVO_committee_support":float(detail["support"][i]),
             "LOVO_p_bar":float(detail["p_bar"][i])}
        for j,name in enumerate(MEMBERS): row["p_"+name]=float(probabilities[i,j])
        sample.append(row)
    write_csv(args.output/"route_metrics.csv",metrics)
    write_csv(args.output/"per_fold_metrics.csv",per_fold)
    write_csv(args.output/"sample_scores.csv",sample)
    write_csv(args.output/"calibration_audit.csv",audit)
    write_csv(args.output/"paired_error_capture.csv",[paired])
    protocol={"source_only":True,"source_rows":799,"year":2017,"future_rows_loaded":0,
              "budget":args.budget,"members":list(MEMBERS),"classifier":"L2 Logistic Regression",
              "calibration":"outer-train four-fold OOF Platt sigmoid",
              "feature_names":canonical_names,"feature_groups":{k:sorted(v) for k,v in mv.EXPECTED_GROUPS.items()},
              "primary_metric":"Stage-I error recall at equal 25% budget",
              "no_stage2_training":True}
    (args.output/"protocol.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    by={r["method"]:r for r in metrics}
    lines=["CARE-Droid strict-2017 leave-one-view-out DI router audit v1","",
           "Same Global24 extraction; four L2-LR members each omit one semantic feature group.",
           "Strict outer-fold OOF with outer-train inner-OOF Platt calibration. No future row or Stage-II training.",""]
    for name in ("A25","Current_DI25","LOVO_ViewDI25"):
        r=by[name];lines.append(f"{name}: routed={r['routed_n']} captured_errors={r['captured_error_n']}/{r['base_error_n']} error_recall={r['error_recall']:.6f} precision={r['routing_precision']:.6f} highconf_error_recall={r['highconf_error_recall']:.6f}")
    lines += ["",f"LOVO-only errors={paired['LOVO_ViewDI_only_error']} Current-DI-only errors={paired['current_DI_only_error']} capture_delta={paired['capture_delta']} exact_p={paired['exact_p_value']:.6f}",
              "PRIMARY_ERROR_CAPTURE_DECISION="+("MORE_ERRORS_CAPTURED" if paired["more_stage1_errors_captured"] else "NOT_MORE_ERRORS_CAPTURED"),
              "A larger error count is promising routing evidence, not proof of end-to-end improvement; repairability and Broken require a frozen Stage-II follow-up."]
    (args.output/"REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("VIEWDI_SOURCE_AUDIT_DONE")
    print("\n".join(lines))

if __name__=="__main__": main()

