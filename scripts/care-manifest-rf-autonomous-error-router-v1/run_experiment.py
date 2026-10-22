#!/usr/bin/env python3
"""Train and evaluate a 2017-only autonomous Manifest135 error router.

The router uses only frozen LR/RF/XGB/SVM Manifest probabilities and derived
confidence/disagreement features. It predicts whether the RF stage-I decision
is wrong and routes when the predicted error probability is at least 0.5. The
threshold is fixed a priori; no route budget is imposed. Source predictions are
strict five-fold OOF. Future models are fit on all 2017 OOF rows and applied to
2018-2021 only for frozen development evaluation.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("/home/boyuanjie/projects/tasa-droid")
OUT = ROOT / "outputs/care_manifest_rf_autonomous_error_router_v1"
FOUR = ROOT / "outputs/care_manifest_four_member_disagreement_v1/run_20261019_pilot01"
COMBINED = ROOT / "outputs/care_manifest_cfg_witness_fixed_source_v2"
M1_FREEZE = ROOT / "outputs/care_manifest_rf_m1_freeze_v1"
SEEDS = [20261018, 20261019, 20261020]
MEMBERS = ["LR", "RF", "XGB", "SVM"]
STRATEGIES = ["STAGE1_ONLY", "ORIGINAL_OTHER3_OP25_REF", "AUTO_ERROR_LR", "ALL_TO_M1"]


def jenc(value):
    if isinstance(value, dict): return {str(k): jenc(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jenc(v) for v in value]
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return None if not np.isfinite(value) else float(value)
    return value


def atomic_json(path, value):
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(jenc(value), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def atomic_csv(path, frame, compression=None):
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, compression=compression)
    os.replace(temp, path)


def atomic_pickle(path, value):
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle: pickle.dump(value, handle, pickle.HIGHEST_PROTOCOL)
    os.replace(temp, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""): h.update(block)
    p = Path(path)
    return {"path": str(p), "size": p.stat().st_size, "sha256": h.hexdigest()}


def preflight():
    if OUT.exists(): raise RuntimeError(f"refusing overwrite: {OUT}")
    required = [
        FOUR / "per_apk_predictions.csv", FOUR / "protocol.json", FOUR / "asset_integrity_check.json",
        COMBINED / "per_apk_w0_w1.csv.gz", COMBINED / "baseline_integrity_check.json",
        COMBINED / "replay_validation.json", COMBINED / "protocol.json",
        M1_FREEZE / "model_contract.json", M1_FREEZE / "baseline_integrity_check.json",
    ]
    required.extend(FOUR / f"windows/{seed}_{fold}.json" for seed in SEEDS for fold in range(6))
    missing = [str(p) for p in required if not p.is_file()]
    if missing: raise RuntimeError("missing assets: " + json.dumps(missing))
    if json.load(open(FOUR / "asset_integrity_check.json")).get("status") != "PASS": raise RuntimeError("four-member integrity")
    if json.load(open(COMBINED / "baseline_integrity_check.json")).get("status") != "PASS": raise RuntimeError("combined integrity")
    if json.load(open(COMBINED / "replay_validation.json")).get("status") != "PASS": raise RuntimeError("combined replay")
    if json.load(open(M1_FREEZE / "baseline_integrity_check.json")).get("status") != "PASS": raise RuntimeError("M1 integrity")
    before = [digest(p) for p in required]
    OUT.mkdir(parents=True, exist_ok=False)
    atomic_json(OUT / "protocol.json", {
        "status": "FROZEN_PROTOCOL",
        "stage1": "Manifest135 RF",
        "stage2": "frozen M1=W1_WITNESS3 probability",
        "router": "StandardScaler + class-weight-balanced LogisticRegression(C=1, liblinear)",
        "router_target": "RF stage-I classification error",
        "router_inputs": "four Manifest probabilities plus deterministic confidence/disagreement transforms",
        "route_rule": "predicted_error_probability >= 0.5",
        "fixed_budget": False,
        "threshold_tuning": False,
        "source_protocol": "2017 five-fold OOF router",
        "future_protocol": "fit all 2017 OOF rows; frozen application to 2018-2021",
        "future_used_for_selection": False,
        "backgrounds_changed": False,
        "prototype_bank_changed": False,
        "direct_c43_changed": False,
    })
    return required, before


def feature_frame(probabilities):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != 4: raise RuntimeError("member probability shape")
    eps = 1e-8
    votes = (p >= 0.5).astype(float)
    rf_vote = votes[:, 1:2]
    pair_diffs = [np.abs(p[:, i] - p[:, j]) for i in range(4) for j in range(i + 1, 4)]
    mean = p.mean(axis=1)
    entropy_rf = -(p[:, 1] * np.log(p[:, 1] + eps) + (1 - p[:, 1]) * np.log(1 - p[:, 1] + eps))
    entropy_mean = -(mean * np.log(mean + eps) + (1 - mean) * np.log(1 - mean + eps))
    other_opposition = np.mean(votes[:, [0, 2, 3]] != rf_vote, axis=1)
    x = np.column_stack([
        p,
        np.abs(p[:, 1] - 0.5),
        mean, p.std(axis=1), p.min(axis=1), p.max(axis=1), p.max(axis=1) - p.min(axis=1),
        votes.sum(axis=1), np.mean(votes != rf_vote, axis=1), other_opposition,
        entropy_rf, entropy_mean,
        *pair_diffs,
    ])
    names = MEMBERS + [
        "rf_margin", "member_mean", "member_std", "member_min", "member_max", "member_range",
        "positive_vote_count", "vote_disagreement_with_rf", "other3_opposition_vote",
        "rf_entropy", "mean_entropy",
    ] + [f"abs_{MEMBERS[i]}_{MEMBERS[j]}" for i in range(4) for j in range(i + 1, 4)]
    if x.shape[1] != len(names): raise RuntimeError("feature name mismatch")
    return x, names


def load_contract():
    combined = pd.read_csv(COMBINED / "per_apk_w0_w1.csv.gz")
    formal = combined[(combined.anchor == "RF") & (combined.strategy == "OTHER3_OP25_REF")].copy()
    if len(formal) != 2788 * 3: raise RuntimeError(f"formal rows {len(formal)}")
    if combined.groupby(["seed", "year", "sha256"]).w1_probability.agg(lambda x: x.max() - x.min()).max() != 0:
        raise RuntimeError("M1 probability spread")
    lookup = formal.set_index(["seed", "sha256"])
    records = []
    feature_names = None
    for seed in SEEDS:
        for fold in range(6):
            window = json.load(open(FOUR / f"windows/{seed}_{fold}.json"))
            ids = list(map(str, window["target_sha"]))
            # The JSON key named members stores target probabilities; model names are in rules.
            if list(window["rules"]) != MEMBERS: raise RuntimeError("rule order changed")
            probs = np.asarray(window["members"], dtype=float)
            x, names = feature_frame(probs)
            feature_names = names if feature_names is None else feature_names
            for sha, row, raw in zip(ids, x, probs):
                key = (seed, sha)
                if key not in lookup.index: raise RuntimeError(f"combined membership missing {key}")
                meta = lookup.loc[key]
                records.append({
                    "seed": seed, "fold": fold, "year": int(meta.year), "sha256": sha, "label": int(meta.label),
                    "p_lr": float(raw[0]), "p_rf": float(raw[1]), "p_xgb": float(raw[2]), "p_svm": float(raw[3]),
                    "m1_probability": float(meta.w1_probability), "original_route": bool(meta.routed),
                    **{f"x_{i}": float(v) for i, v in enumerate(row)},
                })
    frame = pd.DataFrame(records)
    if len(frame) != 2788 * 3 or frame.sha256.nunique() != 2788: raise RuntimeError("window membership")
    return frame, feature_names


def fit_router(frame, feature_names):
    xcols = [f"x_{i}" for i in range(len(feature_names))]
    output = []
    model_info = []
    for seed in SEEDS:
        source = frame[(frame.seed == seed) & (frame.year == 2017)].copy()
        if len(source) != 799 or set(source.fold) != {1, 2, 3, 4, 5}: raise RuntimeError("source fold contract")
        source["stage1_error"] = ((source.p_rf >= 0.5).astype(int) != source.label).astype(int)
        for fold in range(1, 6):
            train = source[source.fold != fold]
            target = source[source.fold == fold]
            model = Pipeline([
                ("scale", StandardScaler()),
                ("lr", LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=2000,
                                          random_state=seed + fold)),
            ])
            model.fit(train[xcols], train.stage1_error)
            score = model.predict_proba(target[xcols])[:, 1]
            for idx, value in zip(target.index, score): output.append((idx, float(value), f"OOF_S{fold}"))
            atomic_pickle(OUT / f"models/{seed}/S{fold}/router.pkl", {
                "pipeline": model, "manifest_seed": seed, "fold": fold, "feature_names": feature_names,
                "training_years": [2017], "target": "RF_error", "threshold": 0.5,
                "train_rows": len(train), "target_rows": len(target), "positive_train_rows": int(train.stage1_error.sum()),
            })
            model_info.append({"seed": seed, "model": f"S{fold}", "train_rows": len(train),
                               "positive_rows": int(train.stage1_error.sum()), "target_rows": len(target)})
        full = Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=2000,
                                      random_state=seed)),
        ])
        full.fit(source[xcols], source.stage1_error)
        future = frame[(frame.seed == seed) & (frame.year > 2017)]
        score = full.predict_proba(future[xcols])[:, 1]
        for idx, value in zip(future.index, score): output.append((idx, float(value), "FULL2017"))
        atomic_pickle(OUT / f"models/{seed}/FULL2017/router.pkl", {
            "pipeline": full, "manifest_seed": seed, "feature_names": feature_names,
            "training_years": [2017], "target": "RF_error", "threshold": 0.5,
            "train_rows": len(source), "positive_train_rows": int(source.stage1_error.sum()),
        })
        model_info.append({"seed": seed, "model": "FULL2017", "train_rows": len(source),
                           "positive_rows": int(source.stage1_error.sum()), "target_rows": len(future)})
    scores = pd.DataFrame(output, columns=["index", "auto_error_probability", "router_model"]).set_index("index")
    frame = frame.join(scores, how="left")
    if frame.auto_error_probability.isna().any(): raise RuntimeError("router prediction missing")
    frame["auto_route"] = frame.auto_error_probability >= 0.5
    return frame, pd.DataFrame(model_info)


def metric_row(y, probability):
    y = np.asarray(y, int); p = np.asarray(probability, float); z = p >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, z, labels=[0, 1]).ravel()
    return {"n": len(y), "BA": balanced_accuracy_score(y, z), "AUC": roc_auc_score(y, p),
            "F1": f1_score(y, z, zero_division=0), "FPR": fp / (fp + tn), "FNR": fn / (fn + tp),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp}


def evaluate(frame):
    perf, apk = [], []
    for row in frame.itertuples(index=False):
        stage1_wrong = int((row.p_rf >= 0.5) != row.label)
        for strategy in STRATEGIES:
            if strategy == "STAGE1_ONLY": routed = False
            elif strategy == "ORIGINAL_OTHER3_OP25_REF": routed = bool(row.original_route)
            elif strategy == "AUTO_ERROR_LR": routed = bool(row.auto_route)
            elif strategy == "ALL_TO_M1": routed = True
            else: raise KeyError(strategy)
            final = row.m1_probability if routed else row.p_rf
            apk.append({"year": row.year, "manifest_seed": row.seed, "fold": row.fold, "sha256": row.sha256,
                        "label": row.label, "strategy": strategy, "routed": routed,
                        "auto_error_probability": row.auto_error_probability, "router_model": row.router_model,
                        "stage1_probability": row.p_rf, "m1_probability": row.m1_probability,
                        "final_probability": final, "stage1_wrong": stage1_wrong,
                        "stage1_correct": 1-stage1_wrong, "m1_correct": int((row.m1_probability >= .5) == row.label),
                        "final_correct": int((final >= .5) == row.label)})
    apk = pd.DataFrame(apk)
    for (year, seed, strategy), q in apk.groupby(["year", "manifest_seed", "strategy"]):
        base_correct = q.stage1_correct.astype(bool); final_correct = q.final_correct.astype(bool)
        errors = q.stage1_wrong.astype(bool)
        perf.append({"year": year, "manifest_seed": seed, "strategy": strategy,
                     "route_rate": q.routed.mean(), "routed_apks": int(q.routed.sum()),
                     "stage1_errors": int(errors.sum()), "routed_stage1_errors": int((q.routed & errors).sum()),
                     "stage1_error_coverage": float((q.routed & errors).sum() / max(errors.sum(), 1)),
                     "routed_error_precision": float((q.routed & errors).sum() / max(q.routed.sum(), 1)),
                     "Fixed": int((~base_correct & final_correct).sum()),
                     "Broken": int((base_correct & ~final_correct).sum()),
                     "Net_Fixed": int((~base_correct & final_correct).sum() - (base_correct & ~final_correct).sum()),
                     **metric_row(q.label, q.final_probability)})
    perf = pd.DataFrame(perf)
    mean = perf.groupby(["year", "strategy"], as_index=False).agg({
        "route_rate": "mean", "routed_apks": "mean", "stage1_errors": "mean",
        "routed_stage1_errors": "mean", "stage1_error_coverage": "mean", "routed_error_precision": "mean",
        "Fixed": "mean", "Broken": "mean", "Net_Fixed": "mean",
        "BA": "mean", "AUC": "mean", "F1": "mean", "FPR": "mean", "FNR": "mean",
    })
    return apk, perf, mean


def finalize(required, before, frame, feature_names, model_info, apk, perf, mean):
    atomic_csv(OUT / "router_training_summary.csv", model_info)
    atomic_csv(OUT / "per_apk_predictions.csv.gz", apk, compression="gzip")
    atomic_csv(OUT / "performance_per_seed.csv", perf)
    atomic_csv(OUT / "performance_seed_mean.csv", mean)
    coverage = mean.pivot(index="year", columns="strategy", values="stage1_error_coverage")
    rate = mean.pivot(index="year", columns="strategy", values="route_rate")
    ba = mean.pivot(index="year", columns="strategy", values="BA")
    comparison = []
    for year in sorted(mean.year.unique()):
        comparison.append({
            "year": int(year),
            "original_route_rate": rate.loc[year, "ORIGINAL_OTHER3_OP25_REF"],
            "auto_route_rate": rate.loc[year, "AUTO_ERROR_LR"],
            "additional_route_rate": rate.loc[year, "AUTO_ERROR_LR"] - rate.loc[year, "ORIGINAL_OTHER3_OP25_REF"],
            "original_error_coverage": coverage.loc[year, "ORIGINAL_OTHER3_OP25_REF"],
            "auto_error_coverage": coverage.loc[year, "AUTO_ERROR_LR"],
            "additional_error_coverage": coverage.loc[year, "AUTO_ERROR_LR"] - coverage.loc[year, "ORIGINAL_OTHER3_OP25_REF"],
            "original_BA": ba.loc[year, "ORIGINAL_OTHER3_OP25_REF"],
            "auto_BA": ba.loc[year, "AUTO_ERROR_LR"],
            "delta_BA": ba.loc[year, "AUTO_ERROR_LR"] - ba.loc[year, "ORIGINAL_OTHER3_OP25_REF"],
            "all_route_BA": ba.loc[year, "ALL_TO_M1"],
        })
    comparison = pd.DataFrame(comparison)
    atomic_csv(OUT / "auto_vs_original_route.csv", comparison)
    source = comparison[comparison.year == 2017].iloc[0]
    future = comparison[comparison.year > 2017]
    decision = {
        "status": "COMPLETE",
        "decision": "AUTO_ROUTER_EVALUATED_NO_FIXED_BUDGET",
        "training_years": [2017], "future_years_development_only": [2018, 2019, 2020, 2021],
        "future_used_for_selection": False, "route_threshold": 0.5, "fixed_budget": False,
        "source_auto_route_rate": source.auto_route_rate,
        "source_original_route_rate": source.original_route_rate,
        "source_auto_error_coverage": source.auto_error_coverage,
        "source_original_error_coverage": source.original_error_coverage,
        "source_auto_BA": source.auto_BA, "source_original_BA": source.original_BA,
        "future_years_with_more_error_coverage": int((future.additional_error_coverage > 0).sum()),
        "future_years_total": len(future),
        "router_inputs_manifest_only": True,
        "stage2_model_changed": False,
    }
    atomic_json(OUT / "final_decision.json", decision)
    lines = [
        "# Manifest135 RF 自主错误路由实验", "",
        "路由器只使用 Manifest135 四成员概率、置信度和分歧。训练仅使用2017；2017为五折 OOF，2018—2021使用全2017冻结路由器。",
        "路由条件固定为 `P(RF第一层出错) >= 0.5`，没有固定路由预算，也没有用未来年份调阈值。", "",
        "| 年份 | 原路由比例 | 自主路由比例 | 原错误覆盖 | 自主错误覆盖 | 覆盖增量 | 原BA | 自主BA | BA增量 | 全路由BA |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in comparison.itertuples(index=False):
        lines.append(f"| {r.year} | {r.original_route_rate:.3f} | {r.auto_route_rate:.3f} | {r.original_error_coverage:.3f} | {r.auto_error_coverage:.3f} | {r.additional_error_coverage:+.3f} | {r.original_BA:.4f} | {r.auto_BA:.4f} | {r.delta_BA:+.4f} | {r.all_route_BA:.4f} |")
    lines += ["", "错误覆盖表示第一层错误中被送入 M1 的比例。自主路由比例是模型自然产生的结果，不是成本约束。",
              "2018—2021仅用于冻结开发评价，不能根据这些年份重新选择阈值、特征或路由器。"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = [digest(p) for p in required]
    changed = [{"before": b, "after": a} for b, a in zip(before, after) if b != a]
    atomic_json(OUT / "asset_integrity_check.json", {"status": "PASS" if not changed else "FAIL",
        "checked_assets": len(required), "changed_assets": changed, "before": before, "after": after})


def main():
    required, before = preflight()
    frame, feature_names = load_contract()
    frame, model_info = fit_router(frame, feature_names)
    apk, perf, mean = evaluate(frame)
    finalize(required, before, frame, feature_names, model_info, apk, perf, mean)
    print("CARE_AUTO_ROUTER_COMPLETE", json.load(open(OUT / "final_decision.json"))["decision"], flush=True)


if __name__ == "__main__": main()
