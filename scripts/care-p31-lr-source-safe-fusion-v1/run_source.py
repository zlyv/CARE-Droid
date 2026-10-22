#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from transforms import ARMS, SELECTABLE_ARMS, normalize_p31, transform


PROJECT = Path(os.environ.get("PLRF_PROJECT", "/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT = Path(os.environ.get("PLRF_OUTPUT", str(PROJECT / "outputs/care-p31-lr-source-safe-fusion-v1"))).resolve()
MEMBER = PROJECT / "artifacts/care-litelocal-stage1-attribution-source-v1/source_member_oof_predictions.csv"
P31 = PROJECT / "artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz"
C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
SEED = 20261009


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    rows = list(rows); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def dump_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def asset(path, purpose):
    stat = Path(path).stat()
    return {"absolute_path": str(Path(path).resolve()), "purpose": purpose,
            "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256(path)}


def auc(y, p):
    y = np.asarray(y, int); p = np.asarray(p, float); pos = int(y.sum()); neg = len(y) - pos
    if not pos or not neg:
        return float("nan")
    order = np.argsort(p, kind="mergesort"); ordered = p[order]; ranks = np.empty(len(p), float); start = 0
    while start < len(p):
        end = start + 1
        while end < len(p) and ordered[end] == ordered[start]: end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0; start = end
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def metrics(y, p):
    y = np.asarray(y, int); p = np.asarray(p, float); pred = p >= 0.5
    tn = int(np.sum((y == 0) & ~pred)); fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & ~pred)); tp = int(np.sum((y == 1) & pred))
    ba = (tn / max(tn + fp, 1) + tp / max(tp + fn, 1)) / 2.0
    bins = np.minimum((p * 10).astype(int), 9); ece = 0.0
    for b in range(10):
        selected = bins == b
        if selected.any(): ece += selected.mean() * abs(p[selected].mean() - y[selected].mean())
    return {"n": len(y), "BA": float(ba), "AUC": auc(y, p),
            "F1": float(2 * tp / max(2 * tp + fp + fn, 1)), "accuracy": float(np.mean(pred == y)),
            "Brier": float(np.mean(np.square(p - y))), "ECE10": float(ece),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp, "errors": fp + fn}


def make_lr(c_value, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(
        C=float(c_value), solver="liblinear", max_iter=5000, random_state=int(seed)))])


def select_c_and_predict(x_train, y_train, x_test, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    splits = list(StratifiedKFold(4, shuffle=True, random_state=int(seed)).split(x_train, y_train))
    candidates = []
    cached = {}
    for c_index, c_value in enumerate(C_GRID):
        oof = np.full(len(y_train), np.nan); fold_ba = []
        for inner, (fit, hold) in enumerate(splits, 1):
            model = make_lr(c_value, seed + 1000 * c_index + inner)
            model.fit(x_train[fit], y_train[fit]); score = model.decision_function(x_train[hold])
            oof[hold] = score; fold_ba.append(metrics(y_train[hold], 1.0 / (1.0 + np.exp(-score)))["BA"])
        value = metrics(y_train, 1.0 / (1.0 + np.exp(-oof)))
        candidates.append({"C": c_value, "inner_BA": value["BA"], "inner_AUC": value["AUC"],
                           "fold_BA_mean": float(np.mean(fold_ba)), "fold_BA_std": float(np.std(fold_ba))})
        cached[c_value] = oof
    chosen = sorted(candidates, key=lambda row: (-row["inner_BA"], row["fold_BA_std"], row["C"]))[0]
    c_value = chosen["C"]; oof = cached[c_value]
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000, random_state=seed + 90000)
    calibrator.fit(oof.reshape(-1, 1), y_train)
    model = make_lr(c_value, seed + 99000); model.fit(x_train, y_train)
    test_score = model.decision_function(x_test)
    probability = calibrator.predict_proba(test_score.reshape(-1, 1))[:, 1]
    return np.clip(probability, 1e-8, 1 - 1e-8), chosen, candidates


def exact_mcnemar(a, b):
    n = int(a + b)
    if n == 0: return 1.0
    low = min(int(a), int(b))
    return float(min(1.0, 2.0 * sum(math.comb(n, k) for k in range(low + 1)) / (2 ** n)))


def main():
    if OUTPUT.exists() and any(path.name != "logs" for path in OUTPUT.iterdir()):
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: non-empty output exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    before = {"assets": [asset(MEMBER, "formal 2017 folds, labels and member OOF probabilities"),
                         asset(P31, "frozen P31 semantic-role representation")]}
    dump_json(OUTPUT / "baseline_asset_manifest_before.json", before)
    protocol = {"scope": "2017 source-only fixed five-fold OOF", "future_rows_loaded": 0,
        "stage2_run": False, "threshold": 0.5, "outer_folds": [1,2,3,4,5],
        "inner_selection": "four-fold outer-train only", "C_grid": list(C_GRID),
        "primary_metric": "pooled outer-OOF balanced accuracy", "seed": SEED,
        "candidate_family": list(ARMS), "count_summary_is_control_only": True}
    dump_json(OUTPUT / "protocol.json", protocol)
    rows = read_csv(MEMBER)
    sha = np.asarray([str(row["sha256"]).strip().lower() for row in rows])
    y = np.asarray([int(float(row["label"])) for row in rows], int)
    folds = np.asarray([int(float(row["fold"])) for row in rows], int)
    formal_raw = np.asarray([float(row["p_P31__LR"]) for row in rows], float)
    formal_rf = np.asarray([float(row["p_P31__RF"]) for row in rows], float)
    with np.load(P31, allow_pickle=False) as archive:
        pids = [str(value).strip().lower() for value in archive["sha256"]]
        p31 = np.asarray(archive["role_prevalence"], float)
    position = {sid: i for i, sid in enumerate(pids)}
    x = normalize_p31(p31[[position[sid] for sid in sha]])
    if len(y) != 799 or sorted(np.bincount(folds)[1:].tolist()) != [159,160,160,160,160]:
        raise RuntimeError("BLOCKED_BY_CONTRACT: source cohort or folds changed")
    raw_metric = metrics(y, formal_raw); rf_metric = metrics(y, formal_rf)
    if abs(raw_metric["BA"] - 0.8584774436090226) > 1e-12 or abs(rf_metric["BA"] - 0.9273809523809524) > 1e-12:
        raise RuntimeError("BLOCKED_BY_CONTRACT: frozen references did not reproduce")
    predictions = {"B0_FORMAL_RAW": formal_raw, "R_RF_REFERENCE": formal_rf}
    audit_rows = []
    for arm_index, arm in enumerate(ARMS):
        tx = transform(x, arm); probability = np.full(len(y), np.nan)
        for outer in range(1, 6):
            train = np.flatnonzero(folds != outer); test = np.flatnonzero(folds == outer)
            seed = SEED + (arm_index + 1) * 100000 + outer * 1000
            probability[test], chosen, candidates = select_c_and_predict(tx[train], y[train], tx[test], seed)
            for row in candidates:
                audit_rows.append({"arm": arm, "outer_fold": outer, **row,
                                   "selected": int(row["C"] == chosen["C"])})
            print(f"P31_LR_SOURCE_SAFE arm={arm} fold={outer}/5 C={chosen['C']}", flush=True)
        if not np.isfinite(probability).all():
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: incomplete OOF arm={arm}")
        predictions[arm] = probability
    metric_rows = []; fold_rows = []; paired_rows = []
    for arm, probability in predictions.items():
        value = {"arm": arm, "feature_width": 31 if arm == "B0_FORMAL_RAW" else
                 (31 if arm == "R_RF_REFERENCE" else transform(x[:1], arm).shape[1]), **metrics(y, probability)}
        metric_rows.append(value)
        for fold in range(1, 6):
            selected = folds == fold; fold_rows.append({"arm": arm, "fold": fold, **metrics(y[selected], probability[selected])})
        if arm not in ("B0_FORMAL_RAW", "R_RF_REFERENCE"):
            raw_ok = (formal_raw >= .5) == y; arm_ok = (probability >= .5) == y
            only_arm = int(np.sum(arm_ok & ~raw_ok)); only_raw = int(np.sum(raw_ok & ~arm_ok))
            candidate_folds = [r for r in fold_rows if r["arm"] == arm]
            raw_folds = {r["fold"]: r for r in fold_rows if r["arm"] == "B0_FORMAL_RAW"}
            paired_rows.append({"arm": arm, "candidate_only_correct": only_arm, "raw_only_correct": only_raw,
                "paired_net_vs_raw": only_arm - only_raw, "mcnemar_exact_p": exact_mcnemar(only_arm, only_raw),
                "folds_noninferior_to_raw": sum(r["BA"] >= raw_folds[r["fold"]]["BA"] for r in candidate_folds)})
    write_csv(OUTPUT / "source_metrics.csv", metric_rows); write_csv(OUTPUT / "source_fold_metrics.csv", fold_rows)
    write_csv(OUTPUT / "regularization_audit.csv", audit_rows); write_csv(OUTPUT / "paired_vs_raw.csv", paired_rows)
    prediction_rows = []
    for i, sid in enumerate(sha):
        row = {"sha256": sid, "label": int(y[i]), "fold": int(folds[i])}
        row.update({arm: float(probability[i]) for arm, probability in predictions.items()})
        prediction_rows.append(row)
    write_csv(OUTPUT / "source_predictions.csv", prediction_rows)
    eligible = [row for row in metric_rows if row["arm"] in SELECTABLE_ARMS]
    leader = sorted(eligible, key=lambda row: (-row["BA"], -row["AUC"], row["arm"]))[0]
    pair = next(row for row in paired_rows if row["arm"] == leader["arm"])
    criteria = {"BA_gain_vs_raw>=1pp": leader["BA"] >= raw_metric["BA"] + .01,
        "majority_folds_noninferior_to_raw": pair["folds_noninferior_to_raw"] >= 3,
        "paired_p<0.05": pair["mcnemar_exact_p"] < .05,
        "within_2pp_of_RF": leader["BA"] >= rf_metric["BA"] - .02}
    status = "SOURCE_CANDIDATE_FOUND" if all(criteria.values()) else "SOURCE_CANDIDATE_NOT_STRONG_ENOUGH"
    decision = {"status": status, "source_leader": leader["arm"], "source_leader_metrics": leader,
        "formal_raw_LR": raw_metric, "formal_RF_reference": rf_metric, "criteria": criteria,
        "future_rows_loaded": 0, "stage2_run": False,
        "next_gate": "freeze one source-selected candidate before any future diagnostic" if status.endswith("FOUND") else "stop; do not inspect future"}
    dump_json(OUTPUT / "final_decision.json", decision)
    after = [asset(MEMBER, before["assets"][0]["purpose"]), asset(P31, before["assets"][1]["purpose"])]
    changed = [a["absolute_path"] for a, b in zip(after, before["assets"]) if a["sha256"] != b["sha256"] or a["size_bytes"] != b["size_bytes"]]
    dump_json(OUTPUT / "baseline_asset_manifest_after.json", {"assets": after})
    dump_json(OUTPUT / "baseline_integrity_check.json", {"status": "PASS" if not changed else "FAIL", "changed": changed})
    if changed: raise RuntimeError("BLOCKED_BY_CONTRACT: baseline assets changed")
    report = ["# P31-LR source-safe fusion v1", "", f"Status: **{status}**", "",
        "Only 2017 fixed-fold OOF was used. Future rows loaded: 0. Stage-II was not run.", "",
        f"Source leader: `{leader['arm']}`; BA={leader['BA']:.6f}; AUC={leader['AUC']:.6f}; F1={leader['F1']:.6f}.",
        f"Formal raw LR BA={raw_metric['BA']:.6f}; formal RF reference BA={rf_metric['BA']:.6f}.", "",
        "The count-summary arm is a shortcut control and is ineligible for selection."]
    (OUTPUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"P31_LR_SOURCE_SAFE_FUSION_STATUS={status}", flush=True)
    print(f"SOURCE_LEADER={leader['arm']} BA={leader['BA']:.6f} AUC={leader['AUC']:.6f} F1={leader['F1']:.6f}", flush=True)
    print("FUTURE_ROWS_LOADED=0 STAGE2_RUN=false", flush=True)


if __name__ == "__main__":
    main()

