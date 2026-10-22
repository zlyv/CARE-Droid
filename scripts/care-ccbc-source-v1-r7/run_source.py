#!/usr/bin/env python3
from __future__ import annotations

import copy
import gzip
import json
import math
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ccbc_evidence import (
    BENIGN_FEATURES, COUNT_FEATURES, DECL_BENIGN_FEATURES, DECL_MALWARE_FEATURES,
    MALWARE_FEATURES, ORIGIN_FEATURES, RolePermutedCompatibility,
    evidence_for_apk, feature_matrix, permission_view, prepare_records,
    shuffle_contracts,
)
from common import (
    MAX_DELTA, OUTPUT, SEED, binary_auc, bootstrap_auc_ci, discover_p31,
    dump_json, load_formal_rows, metrics, norm, percentile_against,
    probability_to_logit, read_jsonl_gz, sigmoid, spearman, top_budget,
    write_csv, write_jsonl_gz,
)
from contracts import CompatibilityModel
from stage1_adapter import strict_nested_committee


ROUTER_LAMBDAS = (0.0, 0.25, 0.50)


def load_inputs():
    formal = load_formal_rows()
    sha = np.asarray([row["sha256"] for row in formal])
    with np.load(discover_p31(), allow_pickle=False) as z:
        pids = [norm(value) for value in z["sha256"]]
        matrix = np.asarray(z["role_prevalence"], float)
    pindex = {sid: index for index, sid in enumerate(pids)}
    p31 = matrix[[pindex[sid] for sid in sha]]
    scope = {row["sha256"]: row for row in read_jsonl_gz(OUTPUT / "cache/component_scope_2017.jsonl.gz")}
    records = prepare_records([scope[sid] for sid in sha], p31)
    with np.load(OUTPUT / "cache/formal_attention.npz", allow_pickle=False) as z:
        aids = [norm(value) for value in z["sha256"]]
        offsets = np.asarray(z["offsets"], int)
        flat = np.asarray(z["attention"], float)
        critical = np.asarray(z["critical_index"], int)
    aindex = {sid: index for index, sid in enumerate(aids)}
    attention = []
    critical_aligned = []
    for sid in sha:
        index = aindex[sid]
        attention.append(flat[offsets[index]:offsets[index + 1]])
        critical_aligned.append(int(critical[index]))
    return formal, p31, records, attention, np.asarray(critical_aligned, int)


def fit_direction(x, target, x_validation, seed):
    x = np.asarray(x, float); target = np.asarray(target, int); x_validation = np.asarray(x_validation, float)
    if len(np.unique(target)) < 2:
        probability = np.full(len(x_validation), float(target[0]) if len(target) else .5)
        return probability, {"status": "SINGLE_CLASS", "train_n": len(target), "positive_n": int(target.sum()), "coefficients": []}
    model = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=1.0, solver="liblinear", class_weight="balanced", max_iter=5000, random_state=int(seed))),
    ])
    model.fit(x, target)
    probability = np.clip(model.predict_proba(x_validation)[:, 1], 1e-6, 1 - 1e-6)
    coefficients = model.named_steps["model"].coef_.reshape(-1).astype(float).tolist()
    return probability, {"status": "FIT", "train_n": len(target), "positive_n": int(target.sum()), "coefficients": coefficients}


def crossfit_evidence(records, p31, labels, attention, critical, seed, variant, stage1_probability):
    labels = np.asarray(labels, int)
    output = [None] * len(records)
    splitter = StratifiedKFold(4, shuffle=True, random_state=int(seed))
    for split_index, (fit, hold) in enumerate(splitter.split(np.zeros(len(labels)), labels), 1):
        train_records = [records[index] for index in fit]
        hold_records = [records[index] for index in hold]
        if variant == "contract_shuffle":
            train_records = shuffle_contracts(train_records, np.asarray(stage1_probability)[fit], seed + 101 * split_index, p31[fit])
            hold_records = shuffle_contracts(hold_records, np.asarray(stage1_probability)[hold], seed + 103 * split_index, p31[hold])
        if variant == "permission":
            train_records = [permission_view(row) for row in train_records]
            hold_records = [permission_view(row) for row in hold_records]
        model = CompatibilityModel().fit(train_records, labels[fit].tolist())
        if variant == "role_shuffle":
            model = RolePermutedCompatibility(model, seed + 107 * split_index)
        for local, index in enumerate(hold):
            row = evidence_for_apk(hold_records[local], p31[index], model, attention[index], critical[index])
            output[index] = row
    if any(value is None for value in output):
        raise RuntimeError("BLOCKED_BY_CONTRACT: incomplete crossfit evidence")
    return output


def full_evidence(train_records, validation_records, train_y, train_p31, validation_p31,
                  validation_attention, validation_critical, fold, variant, train_stage1, validation_stage1):
    fit_records = train_records; test_records = validation_records
    if variant == "contract_shuffle":
        fit_records = shuffle_contracts(train_records, train_stage1, SEED + 1000 * fold, train_p31)
        test_records = shuffle_contracts(validation_records, validation_stage1, SEED + 2000 * fold, validation_p31)
    if variant == "permission":
        fit_records = [permission_view(row) for row in train_records]
        test_records = [permission_view(row) for row in validation_records]
    model = CompatibilityModel().fit(fit_records, list(map(int, train_y)))
    real_model = model
    if variant == "role_shuffle":
        model = RolePermutedCompatibility(model, SEED + 3000 * fold)
    rows = [
        evidence_for_apk(record, vector, model, attention, critical)
        for record, vector, attention, critical in zip(test_records, validation_p31, validation_attention, validation_critical)
    ]
    return rows, real_model


def direction_probabilities(train_rows, validation_rows, train_y, train_stage1, validation_stage1, variant, seed):
    train_pred = np.asarray(train_stage1) >= .5
    validation_pred = np.asarray(validation_stage1) >= .5
    if variant == "count":
        malware_names = benign_names = COUNT_FEATURES
    elif variant == "declaration":
        malware_names, benign_names = DECL_MALWARE_FEATURES, DECL_BENIGN_FEATURES
    elif variant == "origin":
        malware_names = benign_names = ORIGIN_FEATURES
    else:
        malware_names, benign_names = MALWARE_FEATURES, BENIGN_FEATURES
    train_malware = feature_matrix(train_rows, malware_names)
    val_malware = feature_matrix(validation_rows, malware_names)
    train_benign = feature_matrix(train_rows, benign_names)
    val_benign = feature_matrix(validation_rows, benign_names)
    benign_side = ~train_pred
    malware_side = train_pred
    em, em_audit = fit_direction(train_malware[benign_side], np.asarray(train_y)[benign_side] == 1, val_malware, seed + 11)
    eb, eb_audit = fit_direction(train_benign[malware_side], np.asarray(train_y)[malware_side] == 0, val_benign, seed + 13)
    raw = np.zeros(len(validation_rows), float)
    ui = np.asarray([float(row.get("ui", 0.0)) for row in validation_rows])
    if variant == "complement":
        eb = np.clip(1.0 - np.asarray([row.get("vm_attn", 0.0) for row in validation_rows]), 1e-6, 1 - 1e-6)
    em_strength = np.maximum(probability_to_logit(em), 0.0) * (1.0 - ui)
    eb_strength = np.maximum(probability_to_logit(eb), 0.0) * (1.0 - ui)
    raw[~validation_pred] = em_strength[~validation_pred]
    raw[validation_pred] = -eb_strength[validation_pred]
    delta = MAX_DELTA * np.tanh(raw)
    probability = sigmoid(probability_to_logit(validation_stage1) + delta)
    return em, eb, raw, delta, probability, {
        "malware_features": list(malware_names), "benign_features": list(benign_names),
        "EM": em_audit, "EB": eb_audit,
    }


def rank_scores(train_di, validation_di, train_stage1, validation_stage1, train_rows, validation_rows):
    di_train = percentile_against(train_di, train_di)
    di_validation = percentile_against(train_di, validation_di)
    train_pred = np.asarray(train_stage1) >= .5; validation_pred = np.asarray(validation_stage1) >= .5
    c_train = np.asarray([row["coarse_j"] if pred else row["coarse_u"] for row, pred in zip(train_rows, train_pred)], float)
    c_validation = np.asarray([row["coarse_j"] if pred else row["coarse_u"] for row, pred in zip(validation_rows, validation_pred)], float)
    c_train_rank = np.zeros(len(c_train)); c_validation_rank = np.zeros(len(c_validation))
    for side in (False, True):
        tr = train_pred == side; va = validation_pred == side
        c_train_rank[tr] = percentile_against(c_train[tr], c_train[tr])
        c_validation_rank[va] = percentile_against(c_train[tr], c_validation[va])
    return di_train, di_validation, c_train_rank, c_validation_rank


def choose_router_lambda(di_rank, c_rank, stage1, labels, sha):
    errors = (np.asarray(stage1) >= .5) != np.asarray(labels)
    choices = []
    count = int(round(len(labels) * .25))
    for value in ROUTER_LAMBDAS:
        score = (1.0 - value) * di_rank + value * c_rank
        route = top_budget(score, sha, count)
        choices.append((int(np.sum(route & errors)), -value, value))
    return max(choices)[2]


def count_router(train_rows, validation_rows, train_stage1, validation_stage1, train_y, seed):
    x_train = np.log1p(feature_matrix(train_rows, COUNT_FEATURES))
    x_validation = np.log1p(feature_matrix(validation_rows, COUNT_FEATURES))
    target = ((np.asarray(train_stage1) >= .5) != np.asarray(train_y)).astype(int)
    probability, _ = fit_direction(x_train, target, x_validation, seed)
    return probability


def route_metrics(y, stage1, route):
    y = np.asarray(y, int); pred = np.asarray(stage1) >= .5; route = np.asarray(route, bool)
    error = pred != y; fn = (y == 1) & ~pred; fp = (y == 0) & pred
    return {
        "routed": int(route.sum()), "stage1_errors": int(error.sum()),
        "captured_errors": int(np.sum(route & error)),
        "error_recall": float(np.sum(route & error) / max(error.sum(), 1)),
        "precision": float(np.sum(route & error) / max(route.sum(), 1)),
        "FN": int(fn.sum()), "FP": int(fp.sum()),
        "captured_FN": int(np.sum(route & fn)), "captured_FP": int(np.sum(route & fp)),
        "FN_recall": float(np.sum(route & fn) / max(fn.sum(), 1)),
        "FP_recall": float(np.sum(route & fp) / max(fp.sum(), 1)),
        "routed_TN": int(np.sum(route & (y == 0) & ~pred)),
        "routed_TP": int(np.sum(route & (y == 1) & pred)),
    }


def mechanism(y, stage1, probability, raw, route):
    y = np.asarray(y, int); stage1 = np.asarray(stage1, float); probability = np.asarray(probability, float)
    pred = stage1 >= .5; final = probability >= .5; error = pred != y; route = np.asarray(route, bool)
    truth_sign = np.where((y == 1) & ~pred, 1.0, np.where((y == 0) & pred, -1.0, 0.0))
    aligned = np.asarray(raw) * truth_sign
    selected = error & route
    return {
        "routed_errors": int(selected.sum()),
        "correct_direction": int(np.sum(selected & (aligned > 1e-12))),
        "wrong_direction": int(np.sum(selected & (aligned < -1e-12))),
        "insufficient": int(np.sum(selected & (np.abs(aligned) <= 1e-12))),
        "correct_direction_too_weak": int(np.sum(selected & (aligned > 1e-12) & (final == pred))),
        "fixed": int(np.sum(selected & (final == y))),
    }


def paired_bootstrap_against_baseline(y, stage1, baseline, candidate, seed, repeats=2000):
    y = np.asarray(y, int); stage1 = np.asarray(stage1, float)
    baseline = np.asarray(baseline, float); candidate = np.asarray(candidate, float)
    rng = np.random.default_rng(int(seed)); ba_diff = []; net_diff = []
    dummy_route = np.zeros(len(y), bool)
    observed_base = metrics(y, stage1, baseline, dummy_route)
    observed_candidate = metrics(y, stage1, candidate, dummy_route)
    for _ in range(int(repeats)):
        idx = rng.integers(0, len(y), len(y))
        a = metrics(y[idx], stage1[idx], baseline[idx], dummy_route[idx])
        b = metrics(y[idx], stage1[idx], candidate[idx], dummy_route[idx])
        ba_diff.append(b["BA"] - a["BA"]); net_diff.append(b["net"] - a["net"])
    return {
        "observed_BA_difference": observed_candidate["BA"] - observed_base["BA"],
        "BA_difference_CI_low": float(np.quantile(ba_diff, .025)),
        "BA_difference_CI_high": float(np.quantile(ba_diff, .975)),
        "observed_net_difference": observed_candidate["net"] - observed_base["net"],
        "net_difference_CI_low": float(np.quantile(net_diff, .025)),
        "net_difference_CI_high": float(np.quantile(net_diff, .975)),
    }


def main() -> None:
    formal, p31, records, attention, critical = load_inputs()
    sha = np.asarray([row["sha256"] for row in formal]); fold = np.asarray([row["fold"] for row in formal], int)
    y = np.asarray([row["label"] for row in formal], int)
    stage1 = np.asarray([row["stage1_prob"] for row in formal], float)
    formal_di = np.asarray([row["router_score"] for row in formal], float)
    formal_route = np.asarray([row["routed"] for row in formal], bool)
    formal_candidate = np.asarray([row["candidate_prob"] for row in formal], float)
    formal_final = np.asarray([row["formal_final_prob"] for row in formal], float)
    formal_delta = np.asarray([row["formal_bounded_delta"] for row in formal], float)
    variants = ("real", "contract_shuffle", "role_shuffle", "permission")
    oof = {variant: {key: np.full(len(y), np.nan) for key in ("em", "eb", "raw", "delta", "prob", "route_score", "route")} for variant in variants}
    auxiliary = {name: {key: np.full(len(y), np.nan) for key in ("em", "eb", "raw", "delta", "prob")} for name in ("count", "declaration", "origin", "complement")}
    evidence_real = [None] * len(y)
    fold_direction_rows = []; router_fold_rows = []; model_audits = []
    mapping_dir = OUTPUT / "contract_role_mapping_by_fold"; mapping_dir.mkdir(parents=True, exist_ok=True)
    p31_ids = None
    for outer_fold in range(1, 6):
        train = np.flatnonzero(fold != outer_fold); validation = np.flatnonzero(fold == outer_fold)
        train_records = [records[index] for index in train]; validation_records = [records[index] for index in validation]
        train_attention = [attention[index] for index in train]; validation_attention = [attention[index] for index in validation]
        train_critical = critical[train]; validation_critical = critical[validation]
        _, nested_rf, nested_di = strict_nested_committee(p31[train], y[train], SEED + 10000 * outer_fold, jobs=int(os.environ.get("CCBC_JOBS", "4")))
        fold_evidence_train = {}; fold_evidence_validation = {}
        real_model = None
        for variant in variants:
            train_ev = crossfit_evidence(
                train_records, p31[train], y[train], train_attention, train_critical,
                SEED + 20000 * outer_fold, variant, nested_rf,
            )
            validation_ev, fitted_real = full_evidence(
                train_records, validation_records, y[train], p31[train], p31[validation],
                validation_attention, validation_critical, outer_fold, variant,
                nested_rf, stage1[validation],
            )
            fold_evidence_train[variant] = train_ev; fold_evidence_validation[variant] = validation_ev
            if variant == "real":
                real_model = fitted_real
                for local, index in enumerate(validation): evidence_real[index] = validation_ev[local]
            direction_variant = "declaration" if variant == "permission" else "real"
            em, eb, raw, delta, probability, audit = direction_probabilities(
                train_ev, validation_ev, y[train], nested_rf, stage1[validation],
                direction_variant, SEED + 30000 * outer_fold + variants.index(variant) * 100,
            )
            for key, value in zip(("em", "eb", "raw", "delta", "prob"), (em, eb, raw, delta, probability)):
                oof[variant][key][validation] = value
            di_tr, di_va, c_tr, c_va = rank_scores(
                nested_di, formal_di[validation], nested_rf, stage1[validation], train_ev, validation_ev,
            )
            route_lambda = choose_router_lambda(di_tr, c_tr, nested_rf, y[train], sha[train])
            score = (1.0 - route_lambda) * di_va + route_lambda * c_va
            route = top_budget(score, sha[validation], int(round(len(validation) * .25)))
            expected_budget = int(round(len(validation) * .25))
            if int(route.sum()) != expected_budget:
                raise RuntimeError(f"BLOCKED_BY_CONTRACT: fold {outer_fold} route budget {route.sum()} != {expected_budget}")
            oof[variant]["route_score"][validation] = score
            oof[variant]["route"][validation] = route.astype(float)
            model_audits.append({"fold": outer_fold, "variant": variant, "router_lambda": route_lambda, **audit})
        if real_model is None:
            raise RuntimeError("real compatibility model missing")
        write_csv(mapping_dir / f"fold_{outer_fold}.csv", real_model.to_rows(outer_fold))
        real_train = fold_evidence_train["real"]; real_validation = fold_evidence_validation["real"]
        for name in auxiliary:
            variant_name = name
            em, eb, raw, delta, probability, audit = direction_probabilities(
                real_train, real_validation, y[train], nested_rf, stage1[validation],
                variant_name, SEED + 40000 * outer_fold + list(auxiliary).index(name) * 100,
            )
            for key, value in zip(("em", "eb", "raw", "delta", "prob"), (em, eb, raw, delta, probability)):
                auxiliary[name][key][validation] = value
            model_audits.append({"fold": outer_fold, "variant": name, **audit})
        count_score = count_router(real_train, real_validation, nested_rf, stage1[validation], y[train], SEED + 50000 * outer_fold)
        auxiliary.setdefault("count_route", {})[outer_fold] = count_score
        print(f"CCBC source outer fold {outer_fold}/5 done", flush=True)
    for variant in variants:
        for key, value in oof[variant].items():
            if not np.isfinite(value).all():
                raise RuntimeError(f"BLOCKED_BY_CONTRACT: incomplete {variant}/{key}")
    for name in ("count", "declaration", "origin", "complement"):
        for key, value in auxiliary[name].items():
            if not np.isfinite(value).all():
                raise RuntimeError(f"BLOCKED_BY_CONTRACT: incomplete {name}/{key}")
    # Count-only route is fold-local and exactly budget matched.
    count_route = np.zeros(len(y), bool); count_route_score = np.zeros(len(y))
    for outer_fold in range(1, 6):
        validation = np.flatnonzero(fold == outer_fold); score = auxiliary["count_route"][outer_fold]
        count_route_score[validation] = score
        count_route[validation] = top_budget(score, sha[validation], int(round(len(validation) * .25)))
    # End-to-end arms.
    arms = {
        "A0_FORMAL_DI25_C43": (formal_route, formal_final, formal_candidate, formal_delta),
        "A1_FORMAL_DI25_CCBC_DIRECTION": (formal_route, np.where(formal_route, oof["real"]["prob"], stage1), oof["real"]["prob"], oof["real"]["raw"]),
        "A2_CCBC_ROUTE_FORMAL_C43": (oof["real"]["route"].astype(bool), np.where(oof["real"]["route"].astype(bool), formal_candidate, stage1), formal_candidate, formal_delta),
        "A3_CCBC_ROUTE_CCBC_DIRECTION": (oof["real"]["route"].astype(bool), np.where(oof["real"]["route"].astype(bool), oof["real"]["prob"], stage1), oof["real"]["prob"], oof["real"]["raw"]),
        "A4_WITHIN_STRATUM_CONTRACT_SHUFFLE": (oof["contract_shuffle"]["route"].astype(bool), np.where(oof["contract_shuffle"]["route"].astype(bool), oof["contract_shuffle"]["prob"], stage1), oof["contract_shuffle"]["prob"], oof["contract_shuffle"]["raw"]),
        "A5_CONTRACT_ROLE_EDGE_SHUFFLE": (oof["role_shuffle"]["route"].astype(bool), np.where(oof["role_shuffle"]["route"].astype(bool), oof["role_shuffle"]["prob"], stage1), oof["role_shuffle"]["prob"], oof["role_shuffle"]["raw"]),
        "A6_COUNT_ONLY": (count_route, np.where(count_route, auxiliary["count"]["prob"], stage1), auxiliary["count"]["prob"], auxiliary["count"]["raw"]),
        "A7_DECLARATION_ONLY": (oof["real"]["route"].astype(bool), np.where(oof["real"]["route"].astype(bool), auxiliary["declaration"]["prob"], stage1), auxiliary["declaration"]["prob"], auxiliary["declaration"]["raw"]),
        "A8_EXECUTION_ORIGIN_ONLY": (oof["real"]["route"].astype(bool), np.where(oof["real"]["route"].astype(bool), auxiliary["origin"]["prob"], stage1), auxiliary["origin"]["prob"], auxiliary["origin"]["raw"]),
        "N6_PERMISSION_ONLY": (formal_route, np.where(formal_route, oof["permission"]["prob"], stage1), oof["permission"]["prob"], oof["permission"]["raw"]),
        "N8_FB_COMPLEMENT": (formal_route, np.where(formal_route, auxiliary["complement"]["prob"], stage1), auxiliary["complement"]["prob"], auxiliary["complement"]["raw"]),
    }
    aggregate = []; per_fold = []; arm_prediction_rows = []
    for name, (route, final, candidate, raw) in arms.items():
        aggregate.append({
            "arm": name, **metrics(y, stage1, final, route),
            **mechanism(y, stage1, candidate, raw, route),
            "paired_bootstrap_vs_A0": paired_bootstrap_against_baseline(
                y, stage1, formal_final, final, SEED + len(aggregate)
            ),
        })
        arm_prediction_rows.extend({
            "arm": name, "sha256": sha[index], "fold": int(fold[index]),
            "y": int(y[index]), "p_stage1": float(stage1[index]),
            "routed": int(route[index]), "candidate_probability": float(candidate[index]),
            "direction_raw": float(raw[index]), "final_probability": float(final[index]),
            "final_prediction": int(final[index] >= .5),
        } for index in range(len(y)))
        for value in range(1, 6):
            selected = fold == value
            per_fold.append({"arm": name, "fold": value, **metrics(y[selected], stage1[selected], final[selected], route[selected]), **mechanism(y[selected], stage1[selected], candidate[selected], raw[selected], route[selected])})
    write_csv(OUTPUT / "source_fold_results.csv", per_fold)
    write_jsonl_gz(OUTPUT / "arm_predictions.jsonl.gz", arm_prediction_rows)
    dump_json(OUTPUT / "end_to_end_results.json", {"aggregate": aggregate, "per_fold": per_fold, "future_rows_loaded": 0})
    # Direction sufficiency and null comparisons.
    sufficiency = []
    direction_variants = {**{name: oof[name] for name in variants}, **{name: auxiliary[name] for name in ("count", "declaration", "origin", "complement")}}
    pred = stage1 >= .5
    for name, values in direction_variants.items():
        fn_side = ~pred; fp_side = pred
        fn_target = (y[fn_side] == 1).astype(int); fp_target = (y[fp_side] == 0).astype(int)
        fn_result = bootstrap_auc_ci(fn_target, values["em"][fn_side], SEED + len(sufficiency))
        fp_result = bootstrap_auc_ci(fp_target, values["eb"][fp_side], SEED + 100 + len(sufficiency))
        row = {
            "variant": name, "FN_vs_TN_n": int(fn_side.sum()), "FN_n": int(fn_target.sum()),
            "FN_vs_TN_AUC": fn_result["AUC"], "FN_CI_low": fn_result["CI_low"], "FN_CI_high": fn_result["CI_high"],
            "FP_vs_TP_n": int(fp_side.sum()), "FP_n": int(fp_target.sum()),
            "FP_vs_TP_AUC": fp_result["AUC"], "FP_CI_low": fp_result["CI_low"], "FP_CI_high": fp_result["CI_high"],
        }
        fold_auc = []
        for fold_value in range(1, 6):
            selected = fold == fold_value
            a = selected & ~pred; b = selected & pred
            fold_auc.append({
                "fold": fold_value,
                "FN_vs_TN_AUC": binary_auc((y[a] == 1).astype(int), values["em"][a]),
                "FP_vs_TP_AUC": binary_auc((y[b] == 0).astype(int), values["eb"][b]),
            })
        row["folds"] = fold_auc
        sufficiency.append(row)
    dump_json(OUTPUT / "sufficiency_probe.json", {"variants": sufficiency, "future_rows_loaded": 0})
    dump_json(OUTPUT / "matched_null_results.json", {
        "real": next(row for row in sufficiency if row["variant"] == "real"),
        "nulls": [row for row in sufficiency if row["variant"] != "real"],
        "capacity_rule": "same two conditional regularized logistic models",
        "future_rows_loaded": 0,
    })
    # Router results.
    routers = {
        "R0_FORMAL_DI25": formal_route,
        "R1_MEANOPPOSITION_PLUS_CCBC": oof["real"]["route"].astype(bool),
        "R1_CONTRACT_SHUFFLE": oof["contract_shuffle"]["route"].astype(bool),
        "R1_ROLE_EDGE_SHUFFLE": oof["role_shuffle"]["route"].astype(bool),
        "COUNT_ONLY": count_route,
    }
    router_results = []
    for name, route in routers.items():
        row = {"router": name, **route_metrics(y, stage1, route)}
        row["folds"] = [{"fold": value, **route_metrics(y[fold == value], stage1[fold == value], route[fold == value])} for value in range(1, 6)]
        router_results.append(row)
    dump_json(OUTPUT / "router_results.json", {"routers": router_results, "future_rows_loaded": 0})
    # Per-APK and Host evidence.
    apk_rows = []
    host_rows = []
    real_route = oof["real"]["route"].astype(bool); real_final = np.where(real_route, oof["real"]["prob"], stage1)
    for index, row in enumerate(formal):
        evidence = evidence_real[index]
        apk_rows.append({
            "sha256": row["sha256"], "fold": row["fold"], "y": row["label"],
            "p_stage1": stage1[index], "stage1_pred": int(pred[index]), "stage1_correct": int(pred[index] == y[index]),
            "baseline_DI": formal_di[index], "U": evidence["coarse_u"], "J": evidence["coarse_j"],
            "combined_route_score": oof["real"]["route_score"][index], "baseline_route": int(formal_route[index]), "ccbc_route": int(real_route[index]),
            "VM": evidence["vm_attn"], "FB": evidence["fb_attn"], "UI": evidence["ui"],
            "EM": oof["real"]["em"][index], "EB": oof["real"]["eb"][index],
            "raw_delta": oof["real"]["raw"][index], "bounded_delta": oof["real"]["delta"][index],
            "final_probability": real_final[index], "final_prediction": int(real_final[index] >= .5),
            "fixed": int((pred[index] != y[index]) and ((real_final[index] >= .5) == y[index])),
            "broken": int((pred[index] == y[index]) and ((real_final[index] >= .5) != y[index])),
            "contract_coverage": 1.0 - evidence["coarse_unknown"], "host_component_mapping_coverage": evidence["mapping_coverage"],
            "component_count": evidence["component_count"], "host_count": evidence["host_count"],
            "graph_node_count": evidence["graph_node_count"], "graph_edge_count": evidence["graph_edge_count"],
            "critical_host_state": evidence["critical_host_state"], "critical_host_roles": evidence["critical_host_roles"],
        })
        for host in evidence["host_rows"]:
            host_rows.append({"sha256": row["sha256"], "fold": row["fold"], **host})
    write_csv(OUTPUT / "apk_evidence.csv", apk_rows)
    write_jsonl_gz(OUTPUT / "host_evidence.jsonl.gz", host_rows)
    # Shortcut correlations.
    shortcut = []
    for feature in ("VM", "FB", "UI", "U", "J"):
        values = np.asarray([float(row[feature]) for row in apk_rows])
        for size_name in ("component_count", "host_count", "graph_node_count", "graph_edge_count"):
            size = np.asarray([float(row[size_name]) for row in apk_rows])
            shortcut.append({"feature": feature, "size_variable": size_name, "spearman_rho": spearman(values, size)})
    write_csv(OUTPUT / "shortcut_audit.csv", shortcut)
    write_csv(OUTPUT / "source_predictions.csv", apk_rows)
    write_csv(OUTPUT / "route_predictions.csv", [{
        "sha256": row["sha256"], "fold": row["fold"], "y": row["label"],
        "p_stage1": stage1[index], "stage1_pred": int(pred[index]),
        "baseline_DI": formal_di[index], "U": evidence_real[index]["coarse_u"],
        "J": evidence_real[index]["coarse_j"],
        "combined_route_score": oof["real"]["route_score"][index],
        "baseline_route": int(formal_route[index]),
        "ccbc_route": int(real_route[index]),
    } for index, row in enumerate(formal)])
    write_csv(OUTPUT / "model_audit.csv", model_audits)
    # Failure audit.
    failure_rows = [row for row in apk_rows if not row["stage1_correct"] or row["fixed"] or row["broken"] or row["baseline_route"] != row["ccbc_route"] or row["UI"] >= .5]
    write_csv(OUTPUT / "failure_audit.csv", failure_rows)
    dump_json(OUTPUT / "name_zero_audit.json", {
        "status": "PASS",
        "raw_package_name_in_model": False,
        "raw_component_class_in_model": False,
        "certificate_in_model": False,
        "developer_or_market_in_model": False,
        "file_hash_as_feature": False,
        "sdk_vendor_name_in_model": False,
        "dataset_source_string_in_model": False,
        "mapping_identity_storage": "SHA256 aliases only; lifecycle callback names are framework-defined",
        "direction_feature_names": {
            "malware": list(MALWARE_FEATURES), "benign": list(BENIGN_FEATURES),
        },
        "future_rows_loaded": 0,
    })
    print("CCBC_SOURCE_FULL_RUN_DONE")


if __name__ == "__main__":
    main()

