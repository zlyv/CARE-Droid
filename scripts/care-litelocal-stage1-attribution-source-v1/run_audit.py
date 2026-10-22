#!/usr/bin/env python3
"""Strict-2017 LiteLocal Stage-I definition and size-attribution audit."""
from __future__ import annotations
import csv
import importlib.util
from pathlib import Path
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from audit_core import (MEMBERS, effective_sample_size, foldwise_top_budget, mean_opposition,
                        metric_row, stage1_probability, stratified_bootstrap_auc_delta)

OUT = Path("artifacts/care-litelocal-stage1-attribution-source-v1")
FEATURES = Path("artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz")
ROUTER = Path("artifacts/care-source-di25-reconstructed-v1/source_router_inputs.csv")
UPSTREAM = Path("scripts/care-litelocal-semantic-source-v1/run_source_oof.py")
ARMS = ("P31", "COOC16", "TRANS16", "FUSED63")
DEFINITIONS = ("LR_ONLY", "MEAN_COMMITTEE", "MEDIAN_COMMITTEE")
NUISANCE = ("apk_bytes", "class_count", "method_ref_count", "sensitive_method_count", "mapped_invoke_count")


def load_upstream():
    spec = importlib.util.spec_from_file_location("litelocal_upstream", UPSTREAM)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def norm(value):
    return str(value or "").strip().casefold()


def diagnostic_lr(seed):
    return Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=1.0, solver="liblinear", max_iter=5000, random_state=seed))])


def assign_train_ecdf_index(train_values, apply_values):
    train = np.asarray(train_values, dtype=float)
    apply = np.asarray(apply_values, dtype=float)
    columns = []
    for j in range(train.shape[1]):
        ordered = np.sort(train[:, j])
        columns.append(np.searchsorted(ordered, apply[:, j], side="right") / max(len(ordered), 1))
    return np.mean(np.column_stack(columns), axis=1)


def main():
    upstream = load_upstream()
    router = read(ROUTER)
    router_map = {norm(r["sha256"]): r for r in router}
    with np.load(FEATURES, allow_pickle=False) as z:
        ids = np.asarray([norm(x) for x in z["sha256"]])
        prevalence = z["role_prevalence"].astype(float)
        cooc = z["within_method_cooccurrence"].astype(float)
        transitions = z["directed_adjacent_transitions"].astype(float)
        audit = z["audit_values"].astype(float)
        audit_names = [str(x) for x in z["audit_names"]]
    y = np.asarray([int(float(router_map[s]["label"])) for s in ids])
    folds = np.asarray([int(float(router_map[s]["fold"])) for s in ids])
    frozen_p = np.asarray([float(router_map[s]["p_stage1"]) for s in ids])
    frozen_route = np.asarray([int(float(router_map[s]["route_DI"])) == 1 for s in ids])
    nuisance = np.log1p(np.maximum(audit[:, [audit_names.index(name) for name in NUISANCE]], 0.0))

    probabilities = {arm: np.full((len(ids), 4), np.nan) for arm in ARMS}
    fold_features = {arm: {} for arm in ARMS}
    fit_audit = []
    size_only = np.full(len(ids), np.nan)
    fused_plus_size = np.full(len(ids), np.nan)
    propensity = np.full(len(ids), np.nan)
    size_index = np.full(len(ids), np.nan)

    for fold in sorted(set(folds.tolist())):
        tr = np.flatnonzero(folds != fold); va = np.flatnonzero(folds == fold)
        # Reproduce the original LiteLocal source protocol exactly.  Only the
        # bootstrap below has its own audit seed; model/SVD/calibration seeds
        # must remain anchored to the original 20260925 experiment.
        seed = 20260925 + 100000 * fold
        for arm in ARMS:
            out, audit_rows, xtr, xva = upstream.committee_nested(arm, prevalence, cooc, transitions, y, tr, va, seed, jobs=4)
            probabilities[arm][va] = out
            fold_features[arm][fold] = (tr, va, xtr, xva)
            fit_audit.extend({"arm": arm, "fold": fold, **row} for row in audit_rows)
        xtr = fold_features["FUSED63"][fold][2]; xva = fold_features["FUSED63"][fold][3]
        size_model = diagnostic_lr(seed + 1); size_model.fit(nuisance[tr], y[tr]); size_only[va] = size_model.predict_proba(nuisance[va])[:, 1]
        combined = diagnostic_lr(seed + 2); combined.fit(np.hstack([xtr, nuisance[tr]]), y[tr]); fused_plus_size[va] = combined.predict_proba(np.hstack([xva, nuisance[va]]))[:, 1]
        propensity_model = diagnostic_lr(seed + 3); propensity_model.fit(nuisance[tr], y[tr]); propensity[va] = propensity_model.predict_proba(nuisance[va])[:, 1]
        size_index[va] = assign_train_ecdf_index(nuisance[tr], nuisance[va])
        print(f"LiteLocal attribution fold {fold}/5 done", flush=True)

    if any(np.isnan(x).any() for x in probabilities.values()) or np.isnan(size_only).any() or np.isnan(fused_plus_size).any():
        raise RuntimeError("BLOCKED_BY_CONTRACT: incomplete OOF predictions")

    prevalence_y = float(y.mean())
    propensity = np.clip(propensity, .05, .95)
    weights = np.where(y == 1, prevalence_y / propensity, (1.0 - prevalence_y) / (1.0 - propensity))
    weights = np.minimum(weights, 10.0)
    rows = []
    route_rows = []
    sample_rows = []
    frozen_error = (frozen_p >= .5) != y
    for arm in ARMS:
        for definition in DEFINITIONS:
            p = stage1_probability(probabilities[arm], definition)
            score = mean_opposition(probabilities[arm], p)
            route = foldwise_top_budget(score, folds, ids)
            error = (p >= .5) != y
            metrics = metric_row(y, p)
            weighted = metric_row(y, p, weights)
            captured = int(np.sum(error & route))
            official = int(np.sum(frozen_error & route))
            rows.append({"arm": arm, "stage1_definition": definition, **metrics,
                         "size_balanced_BA": weighted["BA"], "size_balanced_AUC": weighted["AUC"],
                         "size_balanced_F1": weighted["F1"], "weight_ESS": effective_sample_size(weights)})
            route_rows.append({"arm": arm, "stage1_definition": definition, "routed": int(route.sum()),
                               "own_errors": int(error.sum()), "own_captured": captured,
                               "own_error_recall": float(captured / max(error.sum(), 1)),
                               "routing_precision": float(captured / max(route.sum(), 1)),
                               "frozen_Global24_errors_captured": official,
                               "frozen_Global24_error_recall": float(official / max(frozen_error.sum(), 1)),
                               "route_overlap_with_frozen_DI25": int(np.sum(route & frozen_route))})
    for i, sid in enumerate(ids):
        row = {"sha256": sid, "year": 2017, "fold": int(folds[i]), "label": int(y[i]),
               "p_frozen_Global24": float(frozen_p[i]), "route_frozen_DI25": int(frozen_route[i]),
               "p_size_only": float(size_only[i]), "p_fused_plus_size": float(fused_plus_size[i]),
               "label_propensity_from_size": float(propensity[i]), "size_balance_weight": float(weights[i]),
               "size_index": float(size_index[i])}
        for arm in ARMS:
            for j, member in enumerate(MEMBERS): row[f"p_{arm}__{member}"] = float(probabilities[arm][i, j])
            for definition in DEFINITIONS: row[f"p_{arm}__{definition}"] = float(stage1_probability(probabilities[arm], definition)[i])
        sample_rows.append(row)

    conditional = []
    base = metric_row(y, size_only); combined = metric_row(y, fused_plus_size)
    mean_delta, lo, hi = stratified_bootstrap_auc_delta(y, fused_plus_size, size_only)
    conditional.append({"comparison": "FUSED63_PLUS_SIZE_minus_SIZE_ONLY", "size_only_BA": base["BA"], "size_only_AUC": base["AUC"],
                        "combined_BA": combined["BA"], "combined_AUC": combined["AUC"],
                        "delta_BA": combined["BA"] - base["BA"], "delta_AUC": combined["AUC"] - base["AUC"],
                        "bootstrap_mean_delta_AUC": mean_delta, "bootstrap_CI_low": lo, "bootstrap_CI_high": hi,
                        "semantic_increment_supported": int(lo > 0.0)})

    strata = []
    quintile = np.minimum((size_index * 5).astype(int), 4) + 1
    for q in range(1, 6):
        idx = quintile == q
        for definition in ("LR_ONLY", "MEAN_COMMITTEE"):
            p = stage1_probability(probabilities["FUSED63"], definition)
            pred = p[idx] >= .5
            row = {"size_quintile": q, "stage1_definition": definition, "n": int(idx.sum()),
                   "benign_n": int(np.sum(y[idx] == 0)), "malware_n": int(np.sum(y[idx] == 1)),
                   "BA": float(balanced_accuracy_score(y[idx], pred)), "F1": float(f1_score(y[idx], pred, zero_division=0))}
            row["AUC"] = float(roc_auc_score(y[idx], p[idx])) if len(set(y[idx].tolist())) == 2 else float("nan")
            strata.append(row)

    write(OUT / "stage1_definition_metrics.csv", rows)
    write(OUT / "router_metrics.csv", route_rows)
    write(OUT / "conditional_size_increment.csv", conditional)
    write(OUT / "size_quintile_metrics.csv", strata)
    write(OUT / "source_member_oof_predictions.csv", sample_rows)
    write(OUT / "fit_audit.csv", fit_audit)

    fused_rows = [r for r in rows if r["arm"] == "FUSED63"]
    fused_routes = [r for r in route_rows if r["arm"] == "FUSED63"]
    lines = ["CARE-Droid strict-2017 LiteLocal Stage-I definition and shortcut attribution v1", "",
             "Existing LiteLocal raw features are reused; no APK reanalysis, future row, call graph, or Stage-II training.",
             "The original report used LR_ONLY for classification while all four members contributed to MeanOpposition routing.", "", "FUSED63 Stage-I definitions:"]
    for row in fused_rows:
        rr = next(x for x in fused_routes if x["stage1_definition"] == row["stage1_definition"])
        lines.append(f"{row['stage1_definition']}: BA={row['BA']:.6f} AUC={row['AUC']:.6f} F1={row['F1']:.6f} errors={row['errors']} captured={rr['own_captured']}/{rr['own_errors']} recall25={rr['own_error_recall']:.6f} precision25={rr['routing_precision']:.6f} size_balanced_BA={row['size_balanced_BA']:.6f} size_balanced_AUC={row['size_balanced_AUC']:.6f}")
    c = conditional[0]
    lines += ["", "Conditional size attribution:",
              f"SIZE_ONLY: BA={c['size_only_BA']:.6f} AUC={c['size_only_AUC']:.6f}",
              f"FUSED63+SIZE: BA={c['combined_BA']:.6f} AUC={c['combined_AUC']:.6f}",
              f"incremental DeltaAUC={c['delta_AUC']:+.6f} bootstrap95CI=[{c['bootstrap_CI_low']:+.6f},{c['bootstrap_CI_high']:+.6f}] semantic_increment_supported={bool(c['semantic_increment_supported'])}",
              f"size-balance effective_sample_size={effective_sample_size(weights):.2f}/799", "",
              "Interpretation boundary: size predictability proves nuisance information is encoded; it does not by itself prove that the malware decision uses a size shortcut.",
              "Do not run future evaluation or retrain C43-RQ until the Stage-I definition and conditional attribution are inspected."]
    (OUT / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("LITELOCAL_STAGE1_ATTRIBUTION_SOURCE_DONE")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
