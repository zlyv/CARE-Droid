#!/usr/bin/env python3
"""CARE-Droid 2017 OOF Stage-II error/complementarity read-only audit.

Inputs: frozen Manifest135 member/router OOF predictions, SCBE W0 checkpoints,
prepared Host/RoleGraph caches, and canonical frozen Host IDs. Outputs: the
new audit artifact tree requested by the study protocol. This file never fits
a model, opens an APK, or modifies a formal asset.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from scipy.stats import spearmanr


ROOT = Path("/home/boyuanjie/projects/tasa-droid")
HERE = ROOT / "scripts/care-stage2-error-complementarity-audit-2017-v1"
OUT = ROOT / "artifacts/care_stage2_error_complementarity_audit_2017_v3"
AUTO = ROOT / "outputs/care_manifest_rf_autonomous_error_router_v1"
FOUR = ROOT / "outputs/care_manifest_four_member_disagreement_v1/run_20261019_pilot01"
SCBE = ROOT / "outputs/care_scbe_v1"
MYD = ROOT / "outputs/care_multiyear_development_v1"
SEM = ROOT / "artifacts/care_host_semantic_adapter_2017_future_v1"
FROZEN = ROOT / "artifacts/care_host_method_semantics_feasibility_2017_v2/frozen"
ENDPOINT_CACHE = ROOT / "artifacts/care-droid-formal-diagnostics-v1/cache/endpoint_cache_2788.jsonl.gz"

MANIFEST_SEEDS = (20261018, 20261019, 20261020)
MODEL_SEEDS = (900000, 900001, 900002)
SPLITS = ("S1", "S2", "S3", "S4", "S5")
MEMBERS = ("LR", "RF", "XGB", "SVM")
RISK_BINS = ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0000001))
TOL = 1e-6

sys.path[:0] = [
    str(ROOT / "scripts/care-background-functional-attribution-v1"),
    str(ROOT / "scripts/care-scbe-v1"),
    str(ROOT / "scripts/care-multiyear-development-v1"),
]


def jvalue(x):
    if isinstance(x, dict):
        return {str(k): jvalue(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jvalue(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def write_json(path: Path, value, replace=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jvalue(value), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    if isinstance(rows, pd.DataFrame):
        frame = rows
    else:
        frame = pd.DataFrame(list(rows))
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return {"path": str(path), "size": path.stat().st_size, "sha256": h.hexdigest()}


def status(stage, state="RUNNING", **extra):
    payload = {"status": state, "stage": stage, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra}
    write_json(OUT / "logs/run_status.json", payload, replace=True)
    print("AUDIT_STAGE", stage, state, flush=True)


def metric(y, p):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    z = p >= 0.5
    tn = int(np.sum((y == 0) & ~z)); fp = int(np.sum((y == 0) & z))
    fn = int(np.sum((y == 1) & ~z)); tp = int(np.sum((y == 1) & z))
    tnr = tn / max(tn + fp, 1); tpr = tp / max(tp + fn, 1)
    return {"BA": 0.5 * (tnr + tpr), "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "FPR": 1 - tnr, "FNR": 1 - tpr, "accuracy": float(np.mean(z == y))}


def outcome_group(y, p1, p2, routed):
    if not routed:
        return "G5_UNROUTED"
    c1 = (p1 >= 0.5) == y; c2 = (p2 >= 0.5) == y
    if not c1 and c2: return "G1_FIXED"
    if c1 and not c2: return "G2_BROKEN"
    if not c1 and not c2: return "G3_BOTH_WRONG"
    return "G4_BOTH_CORRECT"


def potential_group(y, p1, p2):
    return outcome_group(y, p1, p2, True)


def transition(y, p1, p2):
    a, b = int(p1 >= .5), int(p2 >= .5)
    if y == 1 and a == 0 and b == 1: return "FN_TO_TP"
    if y == 0 and a == 1 and b == 0: return "FP_TO_TN"
    if y == 1 and a == 1 and b == 0: return "TP_TO_FN"
    if y == 0 and a == 0 and b == 1: return "TN_TO_FP"
    if y == 1 and a == b == 0: return "FN_STAYS_FN"
    if y == 0 and a == b == 1: return "FP_STAYS_FP"
    return "CORRECT_STAYS_CORRECT"


def safe_auc(y, score):
    y = np.asarray(y, int); score = np.asarray(score, float)
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan")


def risk_bin(v):
    for lo, hi in RISK_BINS:
        if lo <= v < hi:
            return f"[{lo:.1f},{min(hi,1):.1f}{']' if hi > 1 else ')'}"
    return "OUTSIDE_ROUTED_RANGE"


def required_assets():
    paths = [
        AUTO / "per_apk_predictions.csv.gz", AUTO / "protocol.json", AUTO / "asset_integrity_check.json",
        FOUR / "protocol.json", FOUR / "asset_integrity_check.json", MYD / "data.npz",
        MYD / "feature_rows.pkl", MYD / "dataset_audit.json", SEM / "config/baseline_manifest.json",
        SEM / "source_2017/oof_predictions.csv", ENDPOINT_CACHE,
        ROOT / "scripts/care-background-functional-attribution-v1/bfa_core.py",
        ROOT / "scripts/care-multiyear-development-v1/train_stage2.py",
        ROOT / "scripts/care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py",
    ]
    paths += [FOUR / f"windows/{seed}_{fold}.json" for seed in MANIFEST_SEEDS for fold in range(1, 6)]
    paths += [SCBE / f"models/{split}/{seed}/M0_REPRODUCED.pkl" for split in SPLITS for seed in MODEL_SEEDS]
    return paths


def preflight_only():
    missing = [str(p) for p in required_assets() if not p.is_file()]
    if missing:
        raise RuntimeError("missing assets: " + repr(missing))
    if OUT.exists():
        raise RuntimeError("output exists; choose a new version, never overwrite: " + str(OUT))
    a = json.loads((AUTO / "asset_integrity_check.json").read_text())
    f = json.loads((FOUR / "asset_integrity_check.json").read_text())
    if a.get("status") != "PASS" or f.get("status") != "PASS":
        raise RuntimeError("upstream integrity is not PASS")
    route = pd.read_csv(AUTO / "per_apk_predictions.csv.gz")
    q = route[(route.year == 2017) & (route.strategy == "AUTO_ERROR_LR")]
    if len(q) != 799 * 3 or q.sha256.nunique() != 799:
        raise RuntimeError("formal route inventory mismatch")
    print("PREFLIGHT_PASS rows=2397 apks=799", flush=True)


def create_asset_manifests():
    status("1_asset_integrity")
    items = [sha256(p) for p in required_assets()]
    frozen = sorted(FROZEN.glob("*.json"))
    if len(frozen) != 799:
        raise RuntimeError(f"expected 799 frozen Host identity files, got {len(frozen)}")
    frozen_items = [sha256(p) for p in frozen]
    manifest = {"formal_assets": items, "canonical_host_identity_files": frozen_items,
                "scope": "2017 OOF frozen caches/checkpoints only; no APK files", "apks_opened": 0}
    write_json(OUT / "config/asset_manifest.json", manifest)
    write_json(OUT / "integrity/asset_sha256_before.json", manifest)
    return manifest


def load_manifest_members():
    records = []
    for seed in MANIFEST_SEEDS:
        for fold in range(1, 6):
            w = json.loads((FOUR / f"windows/{seed}_{fold}.json").read_text())
            if list(w["rules"]) != list(MEMBERS):
                raise RuntimeError("Manifest member order changed")
            ids = list(map(str, w["target_sha"])); probs = np.asarray(w["members"], float)
            if probs.shape != (len(ids), 4):
                raise RuntimeError("Manifest member probability shape")
            for sid, p in zip(ids, probs):
                records.append({"seed": seed, "outer_fold": fold, "sha256": sid,
                                "p_lr": p[0], "p_rf": p[1], "p_xgb": p[2], "p_svm": p[3]})
    frame = pd.DataFrame(records)
    frame = frame[frame.sha256.isin(load_2017_ids())].copy()
    if len(frame) != 2397 or frame.groupby(["seed", "sha256"]).size().max() != 1:
        raise RuntimeError("Manifest OOF alignment failed")
    return frame


def load_2017_ids():
    with np.load(MYD / "data.npz", allow_pickle=False) as z:
        return set(map(str, z["ids"][z["years"] == 2017]))


def load_w0_probabilities():
    with np.load(MYD / "data.npz", allow_pickle=False) as z:
        ids = np.asarray(z["ids"]).astype(str); years = np.asarray(z["years"], int); yall = np.asarray(z["y"], int)
    source_ix = np.flatnonzero(years == 2017); source_ids = ids[source_ix]; source_y = yall[source_ix]
    position = {sid: i for i, sid in enumerate(source_ids)}
    prob = np.full((len(MODEL_SEEDS), len(source_ids)), np.nan, float)
    fold = np.zeros(len(source_ids), int)
    replay_meta = []
    for fi, split in enumerate(SPLITS, 1):
        for si, seed in enumerate(MODEL_SEEDS):
            path = SCBE / f"models/{split}/{seed}/M0_REPRODUCED.pkl"
            with path.open("rb") as f: obj = pickle.load(f)
            if obj.get("split") != split or int(obj.get("seed")) != seed or obj.get("arm") != "M0_REPRODUCED":
                raise RuntimeError("W0 checkpoint identity mismatch " + str(path))
            target = list(map(str, obj["ids"])); values = np.asarray(obj["probability"], float)
            for sid, value in zip(target, values):
                j = position[sid]; prob[si, j] = value; fold[j] = fi
            replay_meta.append({"split": split, "model_seed": seed, "rows": len(target), "path": str(path)})
    if not np.isfinite(prob).all() or set(fold) != {1,2,3,4,5}:
        raise RuntimeError("incomplete W0 OOF probabilities")
    ensemble = prob.mean(axis=0)
    ref = pd.read_csv(SEM / "source_2017/oof_predictions.csv")
    ref = ref[ref.arm == "W0"].drop_duplicates("sha256").set_index("sha256").loc[source_ids]
    d = float(np.max(np.abs(ensemble - ref.w0_probability.to_numpy(float))))
    if d > TOL:
        raise RuntimeError(f"recent formal W0 ensemble mismatch: {d}")
    return source_ids, source_y, fold, prob, ensemble, replay_meta, d


def aligned_predictions():
    status("2_W0_baseline_replay")
    ids, y, folds, seed_prob, w0, replay_meta, w0_ref_diff = load_w0_probabilities()
    labels = pd.DataFrame({"sha256": ids, "true_label": y, "w0_fold": folds})
    members = load_manifest_members().merge(labels, on="sha256", how="inner", validate="many_to_one")
    route = pd.read_csv(AUTO / "per_apk_predictions.csv.gz")
    route = route[(route.year == 2017) & (route.strategy == "AUTO_ERROR_LR")].copy()
    route = route.rename(columns={"manifest_seed":"seed", "fold":"router_fold"})
    keep = ["seed","sha256","router_fold","auto_error_probability","routed","stage1_probability","final_probability"]
    merged = members.merge(route[keep], on=["seed","sha256"], how="inner", validate="one_to_one")
    if len(merged) != 2397:
        raise RuntimeError("aligned row count mismatch")
    if np.max(np.abs(merged.p_rf - merged.stage1_probability)) > TOL:
        raise RuntimeError("RF member and formal Stage-I probability differ")
    if np.any(merged.outer_fold != merged.router_fold) or np.any(merged.outer_fold != merged.w0_fold):
        raise RuntimeError("fold alignment mismatch")
    pmap = {sid: seed_prob[:, i] for i, sid in enumerate(ids)}
    merged["stage2_seed_900000"] = [pmap[s][0] for s in merged.sha256]
    merged["stage2_seed_900001"] = [pmap[s][1] for s in merged.sha256]
    merged["stage2_seed_900002"] = [pmap[s][2] for s in merged.sha256]
    merged["stage2_probability"] = [w0[list(ids).index(s)] for s in merged.sha256]
    merged["rf_prediction"] = (merged.p_rf >= .5).astype(int)
    merged["stage2_prediction"] = (merged.stage2_probability >= .5).astype(int)
    merged["routed"] = merged.routed.astype(bool)
    merged["replayed_final_probability"] = np.where(merged.routed, merged.stage2_probability, merged.p_rf)
    # The recent semantic experiment is the endpoint of record, not the older router table's m1_probability.
    recent = pd.read_csv(SEM / "source_2017/oof_predictions.csv")
    recent = recent[recent.arm == "W0"].rename(columns={"manifest_seed":"seed"})
    check = merged.merge(recent[["seed","sha256","final_probability","w0_probability"]], on=["seed","sha256"], suffixes=("","_recent"))
    final_diff = float(np.max(np.abs(check.replayed_final_probability - check.final_probability_recent)))
    if final_diff > TOL:
        raise RuntimeError(f"formal endpoint final replay mismatch: {final_diff}")
    merged["final_probability"] = merged.replayed_final_probability
    merged["final_prediction"] = (merged.final_probability >= .5).astype(int)
    merged["stage1_wrong"] = (merged.rf_prediction != merged.true_label).astype(int)
    merged["stage2_correct"] = (merged.stage2_prediction == merged.true_label).astype(int)
    merged["final_correct"] = (merged.final_prediction == merged.true_label).astype(int)
    merged["outcome_group"] = [outcome_group(y0,p1,p2,r) for y0,p1,p2,r in zip(merged.true_label,merged.p_rf,merged.stage2_probability,merged.routed)]
    merged["potential_group"] = [potential_group(y0,p1,p2) for y0,p1,p2 in zip(merged.true_label,merged.p_rf,merged.stage2_probability)]
    merged["transition"] = [transition(y0,p1,p2) for y0,p1,p2 in zip(merged.true_label,merged.p_rf,merged.stage2_probability)]
    merged["net_utility"] = merged.potential_group.map({"G1_FIXED":1,"G2_BROKEN":-1,"G3_BOTH_WRONG":0,"G4_BOTH_CORRECT":0}).astype(int)
    p = merged[["p_lr","p_rf","p_xgb","p_svm"]].to_numpy(float); votes = p >= .5
    merged["rf_confidence"] = np.abs(merged.p_rf - .5)
    merged["rf_uncertainty"] = 1 - 2 * merged.rf_confidence
    merged["member_mean"] = p.mean(1); merged["member_std"] = p.std(1)
    merged["member_range"] = p.max(1)-p.min(1)
    merged["vote_disagreement_with_rf"] = np.mean(votes != votes[:,1:2], axis=1)
    merged["other3_opposition_vote"] = np.mean(votes[:,[0,2,3]] != votes[:,1:2], axis=1)
    ordered = ["sha256","outer_fold","seed","true_label","p_lr","p_rf","p_xgb","p_svm",
               "rf_prediction","auto_error_probability","routed","stage2_seed_900000","stage2_seed_900001",
               "stage2_seed_900002","stage2_probability","stage2_prediction","final_probability","final_prediction",
               "stage1_wrong","stage2_correct","final_correct","outcome_group","potential_group","transition",
               "net_utility","rf_confidence","rf_uncertainty","member_mean","member_std","member_range",
               "vote_disagreement_with_rf","other3_opposition_vote"]
    merged = merged[ordered].sort_values(["seed","outer_fold","sha256"]).reset_index(drop=True)
    write_csv(OUT / "predictions/aligned_2017_oof.csv", merged)
    per_apk = []
    for sid, q in merged.groupby("sha256"):
        row = {"sha256":sid,"true_label":int(q.true_label.iloc[0]),"outer_fold":int(q.outer_fold.iloc[0]),
               "routed_seed_count":int(q.routed.sum()),"fixed_seed_count":int((q.outcome_group=="G1_FIXED").sum()),
               "broken_seed_count":int((q.outcome_group=="G2_BROKEN").sum()),"both_wrong_seed_count":int((q.outcome_group=="G3_BOTH_WRONG").sum()),
               "both_correct_seed_count":int((q.outcome_group=="G4_BOTH_CORRECT").sum()),"unrouted_seed_count":int((q.outcome_group=="G5_UNROUTED").sum()),
               "mean_router_error_probability":float(q.auto_error_probability.mean()),"stage2_probability":float(q.stage2_probability.iloc[0])}
        per_apk.append(row)
    write_csv(OUT / "predictions/per_apk_outcomes.csv", per_apk)
    base = []
    for seed,q in merged.groupby("seed"):
        m1=metric(q.true_label,q.p_rf); m2=metric(q.true_label,q.stage2_probability); mf=metric(q.true_label,q.final_probability)
        base.append({"seed":int(seed),"stage1":m1,"stage2_all_apk_theoretical":m2,"end_to_end":mf,
                     "route_rate":float(q.routed.mean()),"routed":int(q.routed.sum())})
    mean_ba=float(np.mean([x["end_to_end"]["BA"] for x in base]))
    if abs(mean_ba - 0.9194653299916458) > 1e-12:
        raise RuntimeError(f"unexpected current endpoint BA {mean_ba}")
    manifest = {"status":"PASS","endpoint":"recent S1/S3-compatible W0 with Manifest135 autonomous route",
                "formal_rule":"each Manifest seed keeps its own OOF Stage-I/router; W0 is mean of model seeds 900000-900002",
                "manifest_seeds":MANIFEST_SEEDS,"stage2_model_seeds":MODEL_SEEDS,"folds":SPLITS,
                "w0_checkpoints":replay_meta,"router_file":str(AUTO/"per_apk_predictions.csv.gz"),
                "recent_endpoint_file":str(SEM/"source_2017/oof_predictions.csv"),"per_seed":base,
                "mean_2017_end_to_end_BA":mean_ba,"w0_reference_max_diff":w0_ref_diff,
                "final_replay_max_diff":final_diff,"historical_approximately_0_9207_endpoint_used":False}
    write_json(OUT / "config/baseline_manifest.json", manifest)
    write_json(OUT / "integrity/replay_check.json", {"status":"PASS","tolerance":TOL,"baseline":manifest})
    return merged, ids, y


def router_analysis(df):
    status("4_router_complementarity")
    rows=[]
    for seed,q in df.groupby("seed"):
        routed=q.routed; err=q.stage1_wrong.astype(bool); correct=~err
        fixed=q.outcome_group.eq("G1_FIXED"); broken=q.outcome_group.eq("G2_BROKEN")
        bothwrong=q.outcome_group.eq("G3_BOTH_WRONG"); bothcorrect=q.outcome_group.eq("G4_BOTH_CORRECT")
        m=metric(q.true_label,q.final_probability)
        counts=q.outcome_group.value_counts()
        row={"seed":seed,"rows":len(q),"independent_apks":q.sha256.nunique(),"routed":int(routed.sum()),"route_rate":routed.mean(),
             "stage1_errors":int(err.sum()),"routed_stage1_errors":int((routed&err).sum()),
             "stage1_error_capture_rate":float((routed&err).sum()/max(err.sum(),1)),
             "repair_rate_among_routed_errors":float(fixed.sum()/max((routed&err).sum(),1)),
             "damage_rate_among_routed_correct":float(broken.sum()/max((routed&correct).sum(),1)),
             "net_fixed_per_routed_apk":float((fixed.sum()-broken.sum())/max(routed.sum(),1)),
             "both_correct_routing_rate":float(bothcorrect.sum()/max(correct.sum(),1)),
             "both_wrong_routing_rate":float(bothwrong.sum()/max(err.sum(),1)),
             "Fixed":int(fixed.sum()),"Broken":int(broken.sum()),"Net_Fixed":int(fixed.sum()-broken.sum()),
             **{g:int(counts.get(g,0)) for g in ["G1_FIXED","G2_BROKEN","G3_BOTH_WRONG","G4_BOTH_CORRECT","G5_UNROUTED"]},**m}
        for name in ["FN_TO_TP","FP_TO_TN","TP_TO_FN","TN_TO_FP"]:
            row[name]=int(((q.transition==name)&routed).sum())
        rows.append(row)
    write_csv(OUT / "predictions/per_seed_outcomes.csv", rows)
    write_csv(OUT / "router/route_error_capture.csv", rows)
    bins=[]
    for (seed,b),q in df[df.routed].assign(risk_bin=df[df.routed].auto_error_probability.map(risk_bin)).groupby(["seed","risk_bin"]):
        counts=q.outcome_group.value_counts(); bins.append({"seed":seed,"risk_bin":b,"n_seed_apk":len(q),"n_apk":q.sha256.nunique(),
            "mean_risk":q.auto_error_probability.mean(),"malware_ratio":q.true_label.mean(),"mean_rf_confidence":q.rf_confidence.mean(),
            "mean_vote_disagreement":q.vote_disagreement_with_rf.mean(),"net_utility_mean":q.net_utility.mean(),
            **{g:int(counts.get(g,0)) for g in ["G1_FIXED","G2_BROKEN","G3_BOTH_WRONG","G4_BOTH_CORRECT"]}})
    write_csv(OUT / "router/route_gain_groups.csv", bins)
    cheap=[]
    features=["auto_error_probability","rf_uncertainty","member_std","member_range","vote_disagreement_with_rf","other3_opposition_vote"]
    for f in features:
        a=df[df.stage1_wrong==1]; c=df[df.stage1_wrong==0]
        cheap.append({"feature":f,"comparison":"FIXED_vs_BOTH_WRONG_all_legal_OOF","n":len(a),
                      "auc":safe_auc(a.potential_group.eq("G1_FIXED"),a[f]),"spearman_net_utility":float(spearmanr(df[f],df.net_utility).statistic)})
        cheap.append({"feature":f,"comparison":"BROKEN_vs_BOTH_CORRECT_all_legal_OOF","n":len(c),
                      "auc":safe_auc(c.potential_group.eq("G2_BROKEN"),c[f]),"spearman_net_utility":float(spearmanr(df[f],df.net_utility).statistic)})
        for g,q in df.groupby("potential_group"):
            cheap.append({"feature":f,"comparison":"group_distribution","group":g,"n":len(q),"median":q[f].median(),"q25":q[f].quantile(.25),"q75":q[f].quantile(.75)})
    write_csv(OUT / "router/cheap_feature_separability.csv", cheap)
    # All APKs have legal OOF W0 probabilities, so a label-informed oracle is reported only as a theoretical bound.
    oracle=[]
    for seed,q in df.groupby("seed"):
        choose=(q.stage1_wrong==1)&(q.stage2_correct==1)
        op=np.where(choose,q.stage2_probability,q.p_rf)
        oracle.append({"seed":seed,"oracle_routed":int(choose.sum()),**metric(q.true_label,op)})
    util={"stage1_error_auc":safe_auc(df.stage1_wrong,df.auto_error_probability),
          "fixed_positive_utility_auc":safe_auc(df.net_utility.eq(1),df.auto_error_probability),
          "broken_negative_utility_auc":safe_auc(df.net_utility.eq(-1),df.auto_error_probability),
          "spearman_risk_vs_net_utility":float(spearmanr(df.auto_error_probability,df.net_utility).statistic),
          "all_799_have_legal_strict_oof_W0":True,"oracle_is_theoretical_not_deployable":True,"oracle":oracle,
          "metric_denominators":"capture: all Stage-I errors; repair: routed Stage-I errors; damage: routed Stage-I-correct; net/routed: all routed"}
    write_json(OUT / "router/route_utility_summary.json", util)
    return rows, util


def load_static_data():
    with np.load(MYD / "data.npz", allow_pickle=False) as z:
        data={k:z[k] for k in z.files}
    with (MYD / "feature_rows.pkl").open("rb") as f:
        data["feature_rows"] = pickle.load(f)
    source=np.flatnonzero(data["years"]==2017); pos={str(data["ids"][i]):int(i) for i in source}
    audit=json.loads((MYD/"dataset_audit.json").read_text()); roles=list(audit["role_names"])
    return data,source,pos,roles


def endpoint_host_edges(wanted, frozen_views):
    out={sid:0 for sid in wanted}
    with gzip.open(ENDPOINT_CACHE,"rt",encoding="utf-8") as f:
        for line in f:
            d=json.loads(line); sid=str(d.get("sha256","")).lower()
            if sid not in wanted or d.get("status")!="OK": continue
            payload=d["payload"]; hosts=set(map(int,frozen_views[sid]["host_nodes"])); ei=payload.get("edge_index",[[],[]])
            out[sid]=sum(1 for a,b in zip(ei[0],ei[1]) if int(a) in hosts and int(b) in hosts and int(a)!=int(b))
    return out


def static_host_rolegraph_features(data, source, pos, roles, frozen_views, direct_edges):
    rows=[]; off=np.asarray(data["host_offsets"],int)
    for sid in sorted(pos):
        i=pos[sid]; raw=np.asarray(data["host_values"][off[i]:off[i+1]],float); real=int(data["real_host_count"][i])
        actual=raw[:real] if real else np.empty((0,raw.shape[1]))
        selfrole=actual[:,:31] if real else np.empty((0,31)); role_count=(selfrole>0).sum(1) if real else np.asarray([])
        rel=actual[:,74:105] if real else np.empty((0,31)); tail=actual[:,31:43] if real else np.empty((0,12))
        s7=np.asarray(data["s7"][i],float); node=s7[:93].reshape(31,3); gg=s7[93:120]; pair=s7[120:585]; inter=s7[585:].reshape(31,31)
        active=np.any(selfrole>0,axis=0) if real else np.zeros(31,bool)
        rows.append({"sha256":sid,"real_host_count":real,"stored_host_slots":len(raw),"valid_host_count":real,
            "no_real_host":int(real==0),"active_role_count":int(active.sum()),"mean_roles_per_host":float(role_count.mean()) if real else 0.,
            "multi_role_host_ratio":float(np.mean(role_count>1)) if real else 0.,"direct_host_call_edges":int(direct_edges.get(sid,0)),
            "relation_nonzero_host_ratio":float(np.mean(np.any(rel!=0,axis=1))) if real else 0.,"relation_nonzero_values":int(np.count_nonzero(rel)),
            "relation_mean_l1":float(np.mean(np.sum(np.abs(rel),axis=1))) if real else 0.,"relation_max_l1":float(np.max(np.sum(np.abs(rel),axis=1))) if real else 0.,
            "tail12_varying_dimensions":int(np.sum(np.ptp(tail,axis=0)>0)) if real>1 else 0,
            "pairraw_nonzero":int(np.count_nonzero(pair)),"pairraw_strength_sum":float(np.sum(np.maximum(pair,0))),"pairraw_max":float(np.max(pair)),
            "interz_positive":int(np.sum(inter>0)),"interz_negative":int(np.sum(inter<0)),"interz_nonzero_ratio":float(np.mean(inter!=0)),
            "interz_mean_abs":float(np.mean(np.abs(inter))),"interz_max_abs":float(np.max(np.abs(inter))),
            "rolegraph_node_nonzero":int(np.count_nonzero(node)),"rolegraph_complexity":int(np.count_nonzero(pair)+np.count_nonzero(inter)),
            "graphglobal27_mean":float(np.mean(gg)),"graphglobal27_std":float(np.std(gg)),"graphglobal27_l1":float(np.sum(np.abs(gg))),
            "candidate_induced_edge_count":int(frozen_views[sid].get("summary",{}).get("candidate_induced_edge_count",0)),
            "candidate_node_count":int(frozen_views[sid].get("summary",{}).get("candidate_node_count",0))})
    return pd.DataFrame(rows)


def replay_diagnostics(data, pos, roles, frozen_views):
    status("5_8_host_rolegraph_background_attention")
    from bfa_core import formal_split
    import torch
    records=[]; replay=[]
    for fi,split in enumerate(SPLITS,1):
        for seed in MODEL_SEEDS:
            pack=formal_split(data,split,seed,device="cuda",train_data=False)
            replay.append({"split":split,"model_seed":seed,"max_probability_difference":pack["replay_diff"],"rows":len(pack["val"])})
            gt=pack["v4"].graph_tensors(pack["graph_val"],pack["dev"]); model=pack["model"]
            with torch.no_grad():
                for j,idx in enumerate(pack["val"]):
                    sid=str(data["ids"][idx]); bag=np.asarray(pack["bags_val"][j],float); raw=np.asarray(pack["raw_val"][j],float)
                    one={k:v[j] for k,v in gt.items()}; region=torch.as_tensor(bag,dtype=torch.float32,device=pack["dev"])
                    cache,_=model.select_local_evidence(one,region); att=cache.attention.detach().cpu().numpy().astype(float); score=cache.score.detach().cpu().numpy().astype(float)
                    if len(att)!=len(bag) or abs(att.sum()-1)>1e-5 or not np.isfinite(att).all(): raise RuntimeError("attention alignment " + sid)
                    real=int(data["real_host_count"][idx]); view=frozen_views[sid]; host_ids=list(map(str,view["host_ids"]))
                    if len(host_ids)!=real: raise RuntimeError("canonical Host count mismatch "+sid)
                    if real==0: host_ids=["EMPTY_PLACEHOLDER"]
                    critical=int(np.argmax(score)); rank=np.lexsort((np.arange(len(att)),-att)); top3=rank[:min(3,len(rank))]; top5=rank[:min(5,len(rank))]
                    raw_entropy=float(-np.sum(att*np.log(np.clip(att,1e-12,1)))); norm_entropy=raw_entropy/math.log(max(len(att),2)); effective=float(np.exp(raw_entropy))
                    bg=bag[:,107:111]; weighted=np.sum(att[:,None]*bg,axis=0)
                    top_roles=set()
                    for k in top5:
                        top_roles.update(np.flatnonzero(raw[k,:31]>0).tolist())
                    c_roles=np.flatnonzero(raw[critical,:31]>0).tolist()
                    rec={"sha256":sid,"outer_fold":fi,"model_seed":seed,"w0_seed_probability":float(pack["old"]["probability"][j]),
                         "critical_host_stable_id":host_ids[critical],"critical_host_roles":"|".join(roles[k] for k in c_roles),
                         "critical_role_count":len(c_roles),"critical_local_score":float(score[critical]),"mean_local_score":float(score.mean()),
                         "max_attention":float(att.max()),"attention_entropy":raw_entropy,"attention_normalized_entropy":norm_entropy,
                         "attention_effective_support":effective,"top3_attention_sum":float(att[top3].sum()),"top5_attention_sum":float(att[top5].sum()),
                         "top5_role_diversity":len(top_roles),"host_slots":len(att)}
                    names=("external","internal","commonness","residual")
                    for k,n in enumerate(names):
                        rec[f"{n}_mean"]=float(np.mean(bg[:,k])); rec[f"{n}_median"]=float(np.median(bg[:,k])); rec[f"{n}_p90"]=float(np.quantile(bg[:,k],.9)); rec[f"{n}_max"]=float(np.max(bg[:,k])); rec[f"{n}_critical"]=float(bg[critical,k]); rec[f"{n}_attention_weighted"]=float(weighted[k])
                    records.append(rec)
            del pack,gt,model
            torch.cuda.empty_cache()
            print("DIAGNOSTIC_REPLAY",split,seed,flush=True)
    r=pd.DataFrame(records)
    if len(r)!=2397 or r.sha256.nunique()!=799: raise RuntimeError("diagnostic replay inventory")
    agg=[]
    numeric=[c for c in r.columns if c not in {"sha256","outer_fold","model_seed","critical_host_stable_id","critical_host_roles"}]
    for sid,q in r.groupby("sha256"):
        row={"sha256":sid,"outer_fold":int(q.outer_fold.iloc[0]),"critical_host_ids":"|".join(q.critical_host_stable_id.astype(str)),
             "critical_host_roles_by_model":"||".join(q.critical_host_roles.astype(str))}
        for c in numeric: row[c]=float(q[c].mean())
        row["critical_host_seed_agreement"] = float(q.critical_host_stable_id.value_counts().max()/len(q))
        agg.append(row)
    return pd.DataFrame(agg), replay


def feature_group_summary(joined, columns, groups=("outcome_group",)):
    rows=[]
    for keys,q0 in joined.groupby(list(groups)):
        if not isinstance(keys,tuple): keys=(keys,)
        q=q0.drop_duplicates("sha256")
        base={g:k for g,k in zip(groups,keys)}; base.update({"n_seed_apk":len(q0),"n_independent_apk":q.sha256.nunique()})
        for c in columns:
            x=pd.to_numeric(q[c],errors="coerce").dropna()
            rows.append({**base,"feature":c,"mean":x.mean(),"median":x.median(),"q25":x.quantile(.25),"q75":x.quantile(.75),"p90":x.quantile(.9),"min":x.min(),"max":x.max()})
    return rows


def stage2_group_outputs(df, static, diag):
    merged=df.merge(static,on="sha256",validate="many_to_one").merge(diag,on=["sha256","outer_fold"],validate="many_to_one")
    host_cols=["real_host_count","valid_host_count","no_real_host","active_role_count","mean_roles_per_host","multi_role_host_ratio","direct_host_call_edges","relation_nonzero_host_ratio","relation_nonzero_values","relation_mean_l1","relation_max_l1","tail12_varying_dimensions","candidate_node_count","candidate_induced_edge_count"]
    graph_cols=["active_role_count","multi_role_host_ratio","pairraw_nonzero","pairraw_strength_sum","pairraw_max","interz_positive","interz_negative","interz_nonzero_ratio","interz_mean_abs","interz_max_abs","rolegraph_node_nonzero","rolegraph_complexity","graphglobal27_mean","graphglobal27_std","graphglobal27_l1"]
    bg_cols=[f"{n}_{s}" for n in ("external","internal","commonness","residual") for s in ("mean","median","p90","max","critical","attention_weighted")]
    att_cols=["critical_role_count","critical_local_score","mean_local_score","max_attention","attention_entropy","attention_normalized_entropy","attention_effective_support","top3_attention_sum","top5_attention_sum","top5_role_diversity","critical_host_seed_agreement"]
    write_csv(OUT/"stage2/host_structure_groups.csv",feature_group_summary(merged,host_cols))
    write_csv(OUT/"stage2/rolegraph_groups.csv",feature_group_summary(merged,graph_cols))
    write_csv(OUT/"stage2/background_groups.csv",feature_group_summary(merged,bg_cols))
    write_csv(OUT/"stage2/attention_groups.csv",feature_group_summary(merged,att_cols))
    hard=merged[((merged.final_correct==0)|(merged.stage2_correct==0)) & merged.outcome_group.ne("G5_UNROUTED")].copy()
    keep=["sha256","seed","outer_fold","true_label","outcome_group","transition","p_rf","stage2_probability","final_probability","auto_error_probability","critical_host_ids","critical_host_roles_by_model"]+host_cols+graph_cols+bg_cols+att_cols
    write_csv(OUT/"stage2/difficult_fn_fp_diagnostics.csv",hard[keep].sort_values(["true_label","outcome_group","sha256","seed"]))
    return merged,host_cols,graph_cols,bg_cols,att_cols


def bootstrap_and_stability(df, merged):
    status("9_statistics_bootstrap")
    rng=np.random.RandomState(20261021); ids_by_class={y:df[df.true_label==y].sha256.unique() for y in (0,1)}; values=[]
    for b in range(1000):
        sampled=np.concatenate([rng.choice(v,len(v),replace=True) for v in ids_by_class.values()]); parts=[]
        for n,sid in enumerate(sampled):
            q=df[df.sha256==sid].copy(); q["boot_instance"]=n; parts.append(q)
        q=pd.concat(parts,ignore_index=True); bas=[]; stage=[]
        for _,s in q.groupby("seed"):
            bas.append(metric(s.true_label,s.p_rf)["BA"]); stage.append(metric(s.true_label,s.final_probability)["BA"])
        routed=q.routed; fixed=q.outcome_group.eq("G1_FIXED"); broken=q.outcome_group.eq("G2_BROKEN")
        values.append({"bootstrap":b,"stage1_BA":np.mean(bas),"final_BA":np.mean(stage),"delta_BA":np.mean(stage)-np.mean(bas),
                       "fixed_per_seed":fixed.sum()/3,"broken_per_seed":broken.sum()/3,"net_fixed_per_seed":(fixed.sum()-broken.sum())/3,
                       "net_fixed_per_routed":(fixed.sum()-broken.sum())/max(routed.sum(),1)})
    boot=pd.DataFrame(values); summary=[]
    for c in boot.columns[1:]: summary.append({"metric":c,"estimate":boot[c].mean(),"ci95_low":boot[c].quantile(.025),"ci95_high":boot[c].quantile(.975),"unit":"independent APK, class-stratified; all seed rows retained as correlated observations"})
    write_csv(OUT/"statistics/paired_bootstrap.csv",summary)
    stab=[]
    for keys,q in df.groupby(["seed","outer_fold"]):
        fixed=q.outcome_group.eq("G1_FIXED"); broken=q.outcome_group.eq("G2_BROKEN")
        stab.append({"seed":keys[0],"outer_fold":keys[1],"n":len(q),"route_rate":q.routed.mean(),"Fixed":fixed.sum(),"Broken":broken.sum(),"Net_Fixed":fixed.sum()-broken.sum(),"stage1_BA":metric(q.true_label,q.p_rf)["BA"],"final_BA":metric(q.true_label,q.final_probability)["BA"]})
    write_csv(OUT/"statistics/fold_seed_stability.csv",stab)
    # Label-free whole-dataset thresholds define overlapping diagnostic attributes; they are not root-cause labels.
    hq=merged.real_host_count.quantile(.25); rq=merged.active_role_count.quantile(.25); rich_h=merged.real_host_count.quantile(.75); rich_r=merged.active_role_count.quantile(.75); conc=merged.top5_attention_sum.quantile(.75)
    merged=merged.copy(); merged["H1_ROUTING_MISMATCH"]=(merged.routed & merged.net_utility.le(0)).astype(int)
    merged["H2_EVIDENCE_LIMITATION"]=((merged.no_real_host==1)|(merged.real_host_count<=hq)|(merged.active_role_count<=rq)).astype(int)
    merged["H3_DECISION_UTILIZATION_LIMITATION"]=(((merged.real_host_count>=rich_h)|(merged.active_role_count>=rich_r)) & (merged.stage2_correct==0) | ((merged.top5_attention_sum>=conc)&(merged.stage2_correct==0))).astype(int)
    merged["H4_UNRESOLVED"]=(merged[["H1_ROUTING_MISMATCH","H2_EVIDENCE_LIMITATION","H3_DECISION_UTILIZATION_LIMITATION"]].sum(1)==0).astype(int)
    hs=[]
    for h in ["H1_ROUTING_MISMATCH","H2_EVIDENCE_LIMITATION","H3_DECISION_UTILIZATION_LIMITATION","H4_UNRESOLVED"]:
        x=merged[h].astype(bool); hs.append({"attribute":h,"seed_apk_count":int(x.sum()),"independent_apk_count":int(merged.loc[x,"sha256"].nunique()),"rate":float(x.mean()),"definition":"predeclared descriptive attribute; overlapping; not causal"})
    for a,b in [("H1_ROUTING_MISMATCH","H2_EVIDENCE_LIMITATION"),("H1_ROUTING_MISMATCH","H3_DECISION_UTILIZATION_LIMITATION"),("H2_EVIDENCE_LIMITATION","H3_DECISION_UTILIZATION_LIMITATION")]:
        x=merged[a].astype(bool)&merged[b].astype(bool); hs.append({"attribute":a+"&"+b,"seed_apk_count":int(x.sum()),"independent_apk_count":int(merged.loc[x,"sha256"].nunique()),"rate":float(x.mean()),"definition":"attribute intersection"})
    write_csv(OUT/"statistics/diagnostic_hypothesis_summary.csv",hs)
    return summary,stab,hs


def decide_and_report(df, merged, route_rows, util, bootstrap, stability):
    # Cautious evidence rules, fixed before looking at future years. These choose a research direction, not a proven cause.
    oracle_ba=float(np.mean([r["BA"] for r in util["oracle"]])); actual_ba=float(np.mean([metric(q.true_label,q.final_probability)["BA"] for _,q in df.groupby("seed")]))
    stage1_ba=float(np.mean([metric(q.true_label,q.p_rf)["BA"] for _,q in df.groupby("seed")]))
    routing_gap=(oracle_ba-actual_ba>=.01 and util["fixed_positive_utility_auc"]<.65)
    def unique_group(g): return merged[merged.outcome_group==g].drop_duplicates("sha256")
    bw=unique_group("G3_BOTH_WRONG"); fx=unique_group("G1_FIXED"); br=unique_group("G2_BROKEN"); bc=unique_group("G4_BOTH_CORRECT")
    sparse_bw=float(np.mean((bw.no_real_host==1)|(bw.real_host_count<=merged.real_host_count.quantile(.25)))) if len(bw) else 0
    sparse_fx=float(np.mean((fx.no_real_host==1)|(fx.real_host_count<=merged.real_host_count.quantile(.25)))) if len(fx) else 0
    evidence_gap=(len(bw)>=15 and sparse_bw-sparse_fx>=.15)
    rich_bw=float(np.mean((bw.real_host_count>=merged.real_host_count.quantile(.75))|(bw.active_role_count>=merged.active_role_count.quantile(.75)))) if len(bw) else 0
    aggregation_gap=(len(bw)>=15 and rich_bw>=.50 and abs(bw.top5_attention_sum.median()-fx.top5_attention_sum.median())>=.05)
    flags=[routing_gap,evidence_gap,aggregation_gap]
    if sum(flags)>1: decision="MIXED_OR_INCONCLUSIVE"
    elif routing_gap: decision="ROUTING_BENEFIT_GAP"
    elif evidence_gap: decision="HOST_EVIDENCE_GAP"
    elif aggregation_gap: decision="HOST_AGGREGATION_GAP"
    else: decision="MIXED_OR_INCONCLUSIVE"
    avg={k:float(np.mean([r[k] for r in route_rows])) for k in ["route_rate","stage1_error_capture_rate","repair_rate_among_routed_errors","damage_rate_among_routed_correct","net_fixed_per_routed_apk","Fixed","Broken","Net_Fixed"]}
    counts={g:int((df.outcome_group==g).sum()) for g in ["G1_FIXED","G2_BROKEN","G3_BOTH_WRONG","G4_BOTH_CORRECT","G5_UNROUTED"]}
    independent={g:int(df.loc[df.outcome_group==g,"sha256"].nunique()) for g in counts}
    q12=("严格嵌套五折的净收益路由最小实验：输入仅保持现有廉价 Manifest 概率、置信度和分歧；训练目标改为折外 W0 相对 RF 的 +1/0/-1 净收益，并与原错误路由、等路由率随机对照、只预测 RF 错误的容量匹配对照比较。不得使用未来年份调节。" if decision=="ROUTING_BENEFIT_GAP" else "NO_JUSTIFIED_NEXT_EXPERIMENT")
    decision_obj={"status":decision,"evidence_flags":{"routing_gap":routing_gap,"host_evidence_gap":evidence_gap,"host_aggregation_gap":aggregation_gap},
                  "actual_BA":actual_ba,"stage1_BA":stage1_ba,"oracle_theoretical_BA":oracle_ba,"oracle_gap":oracle_ba-actual_ba,
                  "utility_auc":util["fixed_positive_utility_auc"],"sparse_both_wrong":sparse_bw,"sparse_fixed":sparse_fx,"rich_both_wrong":rich_bw,
                  "claim_boundary":"research-direction evidence only; no causal root cause and no new model performance claim","Q12":q12}
    write_json(OUT/"reports/final_decision.json",decision_obj)
    report=f"""# CARE-Droid 第二层错误来源与自主路由互补性审计（2017 OOF）

## 结论

最终状态：**{decision}**。本审计只使用 2017 年严格折外预测和冻结 W0 推理，没有训练模型、扫描 APK 或使用未来年份选择方向。该状态表示值得继续验证的方向，不等同于已证明根因。

## 基线锁定

- 第一层：Manifest135 RF；三个 Manifest seed 独立保留 OOF 概率与自主错误路由。
- 路由：`P(RF error)>=0.5`，无固定预算。
- 第二层：W0 / Direct-C43；每个 APK 的正式概率是三个冻结 W0 model seed 的严格折外概率均值。
- 2017 端到端平均 BA：**{actual_ba:.6f}**；Stage-I 平均 BA：**{stage1_ba:.6f}**。
- 概率重放容差：`1e-6`；完整结果见 `integrity/replay_check.json`。

## 必答问题

**Q1. 当前自主路由平均送入多少 APK？** 平均路由率 {avg['route_rate']:.3%}，即每 seed 平均 {np.mean([r['routed'] for r in route_rows]):.1f}/799。seed/fold 差异见 `statistics/fold_seed_stability.csv`。

**Q2. 路由样本的结果组成？** seed-APK 计数为 {counts}；各组独立 APK 覆盖数为 {independent}。独立 APK 数会重叠，因为同一 APK 在三个 seed 中可能属于不同组。

**Q3. 第一层错误捕获与修复？** 平均错误捕获率 {avg['stage1_error_capture_rate']:.3%}；被路由错误中的修复率 {avg['repair_rate_among_routed_errors']:.3%}。平均 Fixed={avg['Fixed']:.2f}、Broken={avg['Broken']:.2f}、Net Fixed={avg['Net_Fixed']:.2f}。

**Q4. 风险越高是否净收益越高？** 路由风险对“可被 W0 修复”的描述性 AUC 为 {util['fixed_positive_utility_auc']:.4f}，与净收益的 Spearman 相关为 {util['spearman_risk_vs_net_utility']:.4f}。风险分箱结果见 `router/route_gain_groups.csv`；这些结果不用于寻找新阈值。

**Q5. 廉价特征能否区分 FIXED、BROKEN、BOTH_WRONG？** 各特征的单变量 AUC与分组分布见 `router/cheap_feature_separability.csv`。它们仅是描述性区分能力，不构成新路由器证据。

**Q6. 剩余 FN/FP 是否主要集中于 Host 少的 APK？** BOTH_WRONG 中稀疏 Host 比例为 {sparse_bw:.3%}，FIXED 中为 {sparse_fx:.3%}。详细的中位数、IQR、P90 与独立 APK 数见 `stage2/host_structure_groups.csv`。

**Q7. 是否存在证据丰富但仍误判的 APK？** BOTH_WRONG 中达到全体 Host 数或活跃 Role 数上四分位的比例为 {rich_bw:.3%}。具体 APK 在 `stage2/difficult_fn_fp_diagnostics.csv`，这只能说明输入并非普遍稀疏，不能证明聚合必然有错。

**Q8. 三背景差异？** External、Internal、Commonness 以及派生 Residual 的 Host 均值、中位数、P90、最大值、Critical Host 值和 Attention 加权均值均在 `stage2/background_groups.csv`。Residual 被明确保留为派生量，不作为独立第四背景解释。

**Q9. Attention 与 Critical Host 是否有关？** 集中度、熵、有效支持数、Top-3/5 权重、Critical Host seed 一致性和角色多样性见 `stage2/attention_groups.csv`。Attention 只是冻结模型关注区域，没有 Host 恶意真值。

**Q10. 现有证据更支持什么方向？** 当前证据裁决为 **{decision}**。证据标志：routing={routing_gap}, evidence={evidence_gap}, aggregation={aggregation_gap}。

**Q11. 是否存在无法解释的剩余错误？** 存在。`statistics/diagnostic_hypothesis_summary.csv` 保留 H4_UNRESOLVED，并允许 H1-H3 重叠；这些是描述属性，不是根因标签。

**Q12. 下一轮唯一最小实验？** {q12}

## 理论上界和边界

全部 799 个 APK 都有合法严格 OOF W0 概率，因此可计算只用于估计空间的标签 Oracle：平均 BA={oracle_ba:.6f}，相对正式端点差 {oracle_ba-actual_ba:.6f}。该 Oracle 不可部署、不可用于定阈值或训练。2018—2021 未参与本次分析。
"""
    (OUT/"reports/final_report.md").write_text(report,encoding="utf-8")
    return decision_obj


def validate_outputs(df, replay):
    required=["config/audit_config.json","config/baseline_manifest.json","config/asset_manifest.json","integrity/asset_sha256_before.json","integrity/replay_check.json","predictions/aligned_2017_oof.csv","predictions/per_seed_outcomes.csv","predictions/per_apk_outcomes.csv","router/route_error_capture.csv","router/route_gain_groups.csv","router/cheap_feature_separability.csv","router/route_utility_summary.json","stage2/host_structure_groups.csv","stage2/rolegraph_groups.csv","stage2/background_groups.csv","stage2/attention_groups.csv","stage2/difficult_fn_fp_diagnostics.csv","statistics/paired_bootstrap.csv","statistics/fold_seed_stability.csv","statistics/diagnostic_hypothesis_summary.csv","reports/final_report.md","reports/final_decision.json"]
    missing=[x for x in required if not (OUT/x).is_file()]
    tests={"aligned_rows":len(df)==2397,"apks":df.sha256.nunique()==799,"three_seeds":set(df.seed)==set(MANIFEST_SEEDS),"one_fold_per_apk":df.groupby("sha256").outer_fold.nunique().max()==1,"all_replays_pass":max(x["max_probability_difference"] for x in replay)<=TOL,"missing_outputs":missing}
    ok=all(v for k,v in tests.items() if k!="missing_outputs") and not missing
    write_json(OUT/"integrity/validation_results.json",{"status":"PASS" if ok else "FAIL","tests":tests,"model_training":False,"APK_files_opened":0,"future_year_labels_read":False})
    if not ok: raise RuntimeError("validation failed "+repr(tests))


def verify_after(before):
    current={x["path"]:sha256(Path(x["path"])) for x in before["formal_assets"]}
    changed=[]
    for old in before["formal_assets"]:
        if current[old["path"]] != old: changed.append({"before":old,"after":current[old["path"]]})
    # Frozen Host identity files are checked as a compact directory digest after their full before-manifest was recorded.
    frozen_changed=[]
    for old in before["canonical_host_identity_files"]:
        now=sha256(Path(old["path"]))
        if now!=old: frozen_changed.append({"before":old,"after":now})
    result={"status":"PASS" if not changed and not frozen_changed else "FAIL","changed":changed,"frozen_changed":frozen_changed,"checked":len(before["formal_assets"])+len(before["canonical_host_identity_files"])}
    write_json(OUT/"integrity/asset_sha256_after.json",result)
    if result["status"]!="PASS": raise RuntimeError("formal assets changed")


def run():
    if not OUT.exists(): raise RuntimeError("run_tmux.sh must create the new owned output directory")
    config={"scope":"2017 strict OOF only","stage1":"Manifest135 RF","router":"AUTO_ERROR_LR >=0.5; no fixed budget","stage2":"frozen W0 Direct-C43 ensemble","manifest_seeds":MANIFEST_SEEDS,"model_seeds":MODEL_SEEDS,"folds":SPLITS,"training":False,"APK_scan":False,"future_year_selection":False,"bootstrap_repeats":1000,"risk_bins":RISK_BINS}
    write_json(OUT/"config/audit_config.json",config)
    before=create_asset_manifests()
    status("3_OOF_alignment")
    df,ids,y=aligned_predictions()
    route_rows,util=router_analysis(df)
    data,source,pos,roles=load_static_data()
    frozen_views={}
    for sid in ids:
        p=FROZEN/f"{sid}.json"
        frozen_views[sid]=json.loads(p.read_text())
        if str(frozen_views[sid].get("sha256"))!=str(sid): raise RuntimeError("frozen Host SHA mismatch")
    direct=endpoint_host_edges(set(ids),frozen_views)
    static=static_host_rolegraph_features(data,source,pos,roles,frozen_views,direct)
    diag,replay=replay_diagnostics(data,pos,roles,frozen_views)
    merged,host_cols,graph_cols,bg_cols,att_cols=stage2_group_outputs(df,static,diag)
    bootstrap,stability,hyp=bootstrap_and_stability(df,merged)
    status("10_final_report")
    decision=decide_and_report(df,merged,route_rows,util,bootstrap,stability)
    validate_outputs(df,replay)
    status("11_final_integrity")
    verify_after(before)
    status("complete","COMPLETE",decision=decision["status"],rows=2397,apks=799)


if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--preflight-only",action="store_true"); args=ap.parse_args()
    preflight_only() if args.preflight_only else run()
