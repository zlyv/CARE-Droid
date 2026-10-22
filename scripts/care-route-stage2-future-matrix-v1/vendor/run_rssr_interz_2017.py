#!/usr/bin/env python3
"""Strict nested 2017 RSSR experiment on PairRaw + directed Inter-Z.

The structure is frozen to the retained source candidate:
Stats120 + PairRaw465 + InterZ961 clipped to [-5,5].  This script compares
JointLogistic, the existing route-conditioned anchored residual, a fixed RSSR
(gamma=1, lambda=0.03), and fully nested RSSR hyperparameter selection.

RSSR stability is recomputed inside every training split.  It uses repeated
stratified 75% subsamples and the exact score proposed in the design:

  g_j^b = sum_i w_i s_ij (y_i-p_i^0) / sum_i w_i
  C_j   = max(#positive(g_j), #negative(g_j)) / B_s
  A_j   = |median_b g_j^b|
  V_j   = MAD_b(g_j^b)
  S_j   = C_j A_j / (A_j + V_j + eps)

The adaptive penalty is lambda * sum_j beta_j^2/(S_j+eps)^gamma.  It is
implemented by an exact feature reparameterization and the existing anchored
L-BFGS optimizer.  No feature is hard deleted.  Only the fixed 2017 cohort is
used for fitting, tuning, routing, and evaluation; no future label is used.

Copy this file and the helper scripts listed in README into ``scripts/`` and
run from the tasa-droid repository root.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_acn_routing_ablation_2017 as acn
import run_allratio70_vs_nodrop76_experiment as audit
import run_anchored_route_conditioned_residual_2017 as anchored
import run_pairz_structure_ablation_2017 as pairz_exp


THRESHOLD = 0.5
VARIANT = "allratio76_ratio52"
STRUCTURE_TAG = "Stats120_PairRaw465_InterZClip5"


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def balanced_accuracy(labels, probability):
    labels = np.asarray(labels, dtype=np.int64)
    prediction = np.asarray(probability, dtype=np.float64) >= THRESHOLD
    return float(0.5 * (
        np.mean(prediction[labels == 1])
        + np.mean(~prediction[labels == 0])
    ))


def stratified_subsample(labels, fraction, rng):
    labels = np.asarray(labels, dtype=np.int64)
    chosen = []
    for value in (0, 1):
        indices = np.flatnonzero(labels == value)
        count = max(1, int(round(float(fraction) * len(indices))))
        count = min(len(indices), count)
        chosen.append(rng.choice(indices, size=count, replace=False))
    result = np.concatenate(chosen)
    result.sort()
    return result


def stability_scores(
    standardized_structure,
    labels,
    base_probability,
    route,
    eta,
    resamples,
    subsample_fraction,
    seed,
    epsilon,
):
    structure = np.asarray(standardized_structure, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    base_probability = np.asarray(base_probability, dtype=np.float64)
    route = np.asarray(route, dtype=bool)
    if structure.ndim != 2 or structure.shape[0] != len(labels):
        raise ValueError("bad stability structure shape")
    if len(base_probability) != len(labels) or len(route) != len(labels):
        raise ValueError("bad stability vector length")
    weights = np.where(route, 1.0, float(eta))
    residual = labels.astype(np.float64) - base_probability
    rng = np.random.RandomState(int(seed))
    association = np.zeros((int(resamples), structure.shape[1]), dtype=np.float64)
    for repeat in range(int(resamples)):
        selected = stratified_subsample(labels, subsample_fraction, rng)
        selected_weight = weights[selected]
        normalizer = float(np.sum(selected_weight))
        if normalizer <= 0.0:
            raise RuntimeError("zero RSSR subsample weight")
        association[repeat] = (
            structure[selected].T
            @ (selected_weight * residual[selected])
            / normalizer
        )
    positive = np.sum(association > 0.0, axis=0)
    negative = np.sum(association < 0.0, axis=0)
    consistency = np.maximum(positive, negative) / float(resamples)
    median = np.median(association, axis=0)
    strength = np.abs(median)
    mad = np.median(np.abs(association - median[None, :]), axis=0)
    epsilon = float(epsilon)
    stability = consistency * strength / (strength + mad + epsilon)
    stability = np.clip(stability, 0.0, 1.0)
    if np.any(~np.isfinite(stability)):
        raise RuntimeError("non-finite RSSR stability score")
    return {
        "association": association,
        "consistency": consistency,
        "strength": strength,
        "mad": mad,
        "stability": stability,
        "epsilon": float(epsilon),
    }


def fit_adaptive_residual(
    standardized_train,
    labels,
    base_probability,
    weights,
    stability,
    stability_epsilon,
    gamma,
    l2,
    max_iterations,
    tolerance,
):
    stability = np.asarray(stability, dtype=np.float64)
    penalty_scale = np.power(stability + float(stability_epsilon), 0.5 * float(gamma))
    adjusted_train = np.asarray(standardized_train, dtype=np.float64) * penalty_scale
    theta, diagnostic = anchored.fit_offset_residual(
        adjusted_train,
        labels,
        base_probability,
        weights,
        l2,
        max_iterations,
        tolerance,
    )
    beta = penalty_scale * theta
    diagnostic = {
        **diagnostic,
        "gamma": float(gamma),
        "l2": float(l2),
        "stability_epsilon": float(stability_epsilon),
        "penalty_scale_min": float(np.min(penalty_scale)),
        "penalty_scale_median": float(np.median(penalty_scale)),
        "penalty_scale_max": float(np.max(penalty_scale)),
        "beta_l2_norm_original_space": float(np.linalg.norm(beta)),
    }
    return beta, diagnostic


def raw_route_data_for_split(
    fit_rows,
    validation_rows,
    fit_labels,
    compact_by_sha,
    compact_schema,
    original_names,
    crossfit_folds,
    seed,
    knn_k,
    budget,
):
    (
        fit_oof_probability,
        fit_ambiguity,
        fit_collision,
        fit_novelty,
    ) = anchored.crossfit_base_and_risks(
        fit_rows,
        fit_labels,
        compact_by_sha,
        compact_schema,
        original_names,
        crossfit_folds,
        seed + 1000,
        seed + 10000,
        knn_k,
    )
    x_fit, x_validation, _ = audit.compact_matrix_variant(
        VARIANT,
        fit_rows,
        validation_rows,
        compact_by_sha,
        compact_schema,
        original_names,
    )
    validation_probability, _, _ = audit.p72.fit_predict(
        x_fit,
        fit_labels,
        x_validation,
        seed=seed,
    )
    validation_collision, validation_novelty, _, _ = acn.neighborhood_risks(
        x_fit,
        fit_labels,
        x_validation,
        knn_k,
    )
    fit_route_data = anchored.calibrated_routes(
        fit_ambiguity,
        fit_collision,
        fit_novelty,
        fit_ambiguity,
        fit_collision,
        fit_novelty,
        budget,
    )
    validation_route_data = anchored.calibrated_routes(
        fit_ambiguity,
        fit_collision,
        fit_novelty,
        acn.ambiguity(validation_probability),
        validation_collision,
        validation_novelty,
        budget,
    )
    return (
        fit_oof_probability,
        validation_probability,
        fit_route_data,
        validation_route_data,
    )


def tune_rssr(
    outer_fold,
    rows,
    labels,
    structure,
    compact_by_sha,
    compact_schema,
    original_names,
    route_tags,
    args,
):
    labels = np.asarray(labels, dtype=np.int64)
    splitter = StratifiedKFold(
        n_splits=int(args.tuning_folds),
        shuffle=True,
        random_state=int(args.tuning_seed + 1000 * outer_fold),
    )
    combinations = [
        (float(gamma), float(l2))
        for gamma in args.gamma_grid
        for l2 in args.l2_grid
    ]
    scores = {
        (route_tag, gamma, l2): []
        for route_tag in route_tags
        for gamma, l2 in combinations
    }
    audit_rows = []
    for tuning_fold, (fit_index, validation_index) in enumerate(
        splitter.split(np.zeros(len(labels)), labels), start=1
    ):
        fit_rows = [rows[i] for i in fit_index]
        validation_rows = [rows[i] for i in validation_index]
        fit_labels = labels[fit_index]
        validation_labels = labels[validation_index]
        (
            fit_oof_probability,
            validation_probability,
            fit_route_data,
            validation_route_data,
        ) = raw_route_data_for_split(
            fit_rows,
            validation_rows,
            fit_labels,
            compact_by_sha,
            compact_schema,
            original_names,
            args.tuning_crossfit_folds,
            args.tuning_seed + 100000 * outer_fold + 1000 * tuning_fold,
            args.knn_k,
            args.budget,
        )
        feature_names = [
            "structure__" + format(i, "04d") for i in range(structure.shape[1])
        ]
        standardized_fit, standardized_validation, _ = audit.p72.standardize_added(
            structure[fit_index], structure[validation_index], feature_names
        )
        for route_tag, route_key in (
            (route_tags[0], "ambiguity_route"),
            (route_tags[1], "acn_route"),
        ):
            fit_route = fit_route_data[route_key]
            validation_route = validation_route_data[route_key]
            stability = stability_scores(
                standardized_fit,
                fit_labels,
                fit_oof_probability,
                fit_route,
                args.eta,
                args.stability_resamples,
                args.subsample_fraction,
                args.stability_seed
                + 1000000 * outer_fold
                + 10000 * tuning_fold
                + (0 if route_tag == route_tags[0] else 1),
                args.stability_epsilon,
            )
            weights = np.where(fit_route, 1.0, float(args.eta))
            for gamma, l2 in combinations:
                beta, diagnostic = fit_adaptive_residual(
                    standardized_fit,
                    fit_labels,
                    fit_oof_probability,
                    weights,
                    stability["stability"],
                    stability["epsilon"],
                    gamma,
                    l2,
                    args.max_iterations,
                    args.optimizer_tolerance,
                )
                enhanced = anchored.sigmoid(
                    anchored.logit(validation_probability)
                    + standardized_validation @ beta
                )
                final_probability = acn.routed_probability(
                    validation_probability,
                    enhanced,
                    validation_route,
                )
                value = balanced_accuracy(validation_labels, final_probability)
                scores[(route_tag, gamma, l2)].append(value)
                audit_rows.append({
                    "outer_fold": int(outer_fold),
                    "tuning_fold": int(tuning_fold),
                    "route": route_tag,
                    "gamma": gamma,
                    "l2": l2,
                    "BA": value,
                    "fit_n": int(len(fit_index)),
                    "validation_n": int(len(validation_index)),
                    "fit_routed_n": int(np.sum(fit_route)),
                    "validation_routed_n": int(np.sum(validation_route)),
                    "stability_mean": float(np.mean(stability["stability"])),
                    "stability_median": float(np.median(stability["stability"])),
                    "eval_delta_max_abs": float(np.max(np.abs(
                        standardized_validation @ beta
                    ))),
                    "optimizer_success": bool(diagnostic["optimizer_success"]),
                })
    selected = {}
    summary_rows = []
    for route_tag in route_tags:
        candidates = []
        for gamma, l2 in combinations:
            values = np.asarray(scores[(route_tag, gamma, l2)], dtype=np.float64)
            row = {
                "outer_fold": int(outer_fold),
                "route": route_tag,
                "gamma": gamma,
                "l2": l2,
                "mean_BA": float(np.mean(values)),
                "min_BA": float(np.min(values)),
                "std_BA": float(np.std(values)),
            }
            summary_rows.append(row)
            candidates.append(row)
        candidates.sort(key=lambda row: (
            -row["mean_BA"],
            -row["min_BA"],
            row["gamma"],
            abs(math.log(row["l2"] / 0.03)),
            row["l2"],
        ))
        selected[route_tag] = candidates[0]
        for row in summary_rows:
            if row["route"] == route_tag:
                row["selected"] = int(
                    row["gamma"] == candidates[0]["gamma"]
                    and row["l2"] == candidates[0]["l2"]
                )
    return selected, summary_rows, audit_rows


def load_structure_reference(path, cohort_rows, raw_structure):
    path = Path(path)
    if not path.exists():
        raise RuntimeError("multiscale feature reference not found: " + str(path))
    reference = np.load(path)
    expected_sha = np.asarray([
        pairz_exp.norm_sha(row["sha256"]) for row in cohort_rows
    ])
    actual_sha = reference["sha256"].astype(str)
    pair_raw = raw_structure[:, 120:]
    result = {
        "path": str(path),
        "sha_order_equal": bool(np.array_equal(expected_sha, actual_sha)),
        "pair_raw_max_abs_difference": float(np.max(np.abs(
            pair_raw - reference["pair_raw"]
        ))),
        "inter_clip_finite": bool(np.isfinite(reference["inter_z_clipped"]).all()),
    }
    if (
        not result["sha_order_equal"]
        or result["pair_raw_max_abs_difference"] > 0.0
        or not result["inter_clip_finite"]
    ):
        raise RuntimeError("invalid multiscale feature reference: " + repr(result))
    structure = np.concatenate([
        raw_structure[:, :120],
        pair_raw,
        np.asarray(reference["inter_z_clipped"], dtype=np.float64),
    ], axis=1)
    if structure.shape != (799, 1546):
        raise RuntimeError("expected retained structure (799,1546), got " + repr(structure.shape))
    return structure, result


def check_previous_predictions(path, manifest, probabilities, routes, route_tags):
    path = Path(path)
    if not path.exists():
        return {"available": False, "path": str(path)}
    rows = read_csv(path)
    by_sha = {pairz_exp.norm_sha(row["sha256"]): row for row in rows}
    columns = {
        "C76": "C76__prob",
        "A20__Joint": "A20__Joint__S7_Stats120_PairRaw465_InterZClip5__prob",
        "A20__Ordinary": (
            "A20__RouteResidual_eta0.25__S7_Stats120_PairRaw465_InterZClip5__prob"
        ),
        "ACN20__Joint": "ACN20__Joint__S7_Stats120_PairRaw465_InterZClip5__prob",
        "ACN20__Ordinary": (
            "ACN20__RouteResidual_eta0.25__S7_Stats120_PairRaw465_InterZClip5__prob"
        ),
    }
    result = {"available": True, "path": str(path), "rows": int(len(rows))}
    for method, column in columns.items():
        differences = []
        for index, row in enumerate(manifest):
            previous = by_sha[pairz_exp.norm_sha(row["sha256"])]
            differences.append(abs(float(previous[column]) - probabilities[method][index]))
        result[method + "__max_abs_probability_difference"] = float(max(differences))
    result["ambiguity_route_mismatches"] = int(sum(
        int(by_sha[pairz_exp.norm_sha(row["sha256"])]["ambiguity_routed"])
        != int(routes[route_tags[0]][index])
        for index, row in enumerate(manifest)
    ))
    result["acn_route_mismatches"] = int(sum(
        int(by_sha[pairz_exp.norm_sha(row["sha256"])]["acn_routed"])
        != int(routes[route_tags[1]][index])
        for index, row in enumerate(manifest)
    ))
    differences = [
        value for key, value in result.items()
        if key.endswith("__max_abs_probability_difference")
    ]
    if max(differences, default=0.0) > 1e-10:
        raise RuntimeError("previous S7 probability reproduction failed: " + repr(result))
    if result["ambiguity_route_mismatches"] or result["acn_route_mismatches"]:
        raise RuntimeError("previous S7 route reproduction failed: " + repr(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path,
        default=Path("artifacts/p1-features/features.csv"),
    )
    parser.add_argument(
        "--schema", type=Path,
        default=Path("artifacts/p1-features/feature_schema.json"),
    )
    parser.add_argument(
        "--cache", type=Path,
        default=Path(
            "artifacts/p25-endpoint-cache-full-cap256/"
            "endpoint_centered_tensor_cache.jsonl.gz"
        ),
    )
    parser.add_argument(
        "--p72-oof", type=Path,
        default=Path(
            "artifacts/p72-verified-dependency-source-ablation/"
            "p72_oof_predictions.csv"
        ),
    )
    parser.add_argument(
        "--multiscale-features", type=Path,
        default=Path(
            "artifacts/clipped-intra-inter-structure-ablation-2017-v1-full/"
            "multiscale_features.npz"
        ),
    )
    parser.add_argument(
        "--previous-predictions", type=Path,
        default=Path(
            "artifacts/clipped-intra-inter-structure-ablation-2017-v1-full/"
            "source_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/rssr-interz-2017-v1"),
    )
    parser.add_argument("--budget", type=float, default=0.20)
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=20260827)
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument("--ordinary-l2", type=float, default=0.03)
    parser.add_argument("--fixed-gamma", type=float, default=1.0)
    parser.add_argument("--fixed-l2", type=float, default=0.03)
    parser.add_argument(
        "--gamma-grid", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0]
    )
    parser.add_argument(
        "--l2-grid", type=float, nargs="+", default=[0.01, 0.03, 0.1]
    )
    parser.add_argument("--tuning-folds", type=int, default=3)
    parser.add_argument("--tuning-crossfit-folds", type=int, default=3)
    parser.add_argument("--tuning-seed", type=int, default=20260827)
    parser.add_argument("--stability-resamples", type=int, default=24)
    parser.add_argument("--subsample-fraction", type=float, default=0.75)
    parser.add_argument("--stability-seed", type=int, default=20260827)
    parser.add_argument("--stability-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--optimizer-tolerance", type=float, default=1e-8)
    args = parser.parse_args()

    if not 0.0 < args.budget < 1.0:
        raise SystemExit("--budget must be in (0,1)")
    if not 0.0 <= args.eta < 1.0:
        raise SystemExit("--eta must be in [0,1)")
    if not 0.5 <= args.subsample_fraction < 1.0:
        raise SystemExit("--subsample-fraction must be in [0.5,1)")
    if args.stability_resamples < 4:
        raise SystemExit("--stability-resamples must be at least 4")
    if args.stability_epsilon <= 0.0:
        raise SystemExit("--stability-epsilon must be positive")
    if args.tuning_folds < 2 or args.tuning_crossfit_folds < 2:
        raise SystemExit("tuning fold counts must be at least 2")
    if any(value < 0.0 for value in args.gamma_grid):
        raise SystemExit("gamma values must be nonnegative")
    if any(value <= 0.0 for value in args.l2_grid):
        raise SystemExit("L2 values must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    (
        manifest,
        cohort_rows,
        _,
        compact_by_sha,
        compact_schema,
        original_names,
    ) = audit.load_inputs(args)
    cache = audit.exp.read_cache(args.cache)
    print("Reconstructing source Stats120+PairRaw465...")
    raw_structure, _ = audit.future.reconstruct_structure(
        cohort_rows, cache, int(audit.exp.ROLE_VOCABULARY_SIZE)
    )
    structure, structure_reference = load_structure_reference(
        args.multiscale_features, cohort_rows, raw_structure
    )

    labels = np.asarray([row["label"] for row in manifest], dtype=np.int64)
    folds = np.asarray([row["fold"] for row in manifest], dtype=np.int64)
    unique_folds = sorted(np.unique(folds))
    n = len(labels)
    budget_tag = str(int(round(100.0 * args.budget)))
    route_tags = ["A" + budget_tag, "ACN" + budget_tag]
    model_tags = ["Joint", "Ordinary", "RSSR_fixed", "RSSR_nested"]
    methods = ["C76"] + [
        route + "__" + model
        for route in route_tags
        for model in model_tags
    ]
    probability = {method: np.full(n, np.nan) for method in methods}
    routes = {route: np.zeros(n, dtype=bool) for route in route_tags}
    risk_outputs = {
        "ambiguity_percentile": np.full(n, np.nan),
        "collision_percentile": np.full(n, np.nan),
        "novelty_percentile": np.full(n, np.nan),
        "acn_score": np.full(n, np.nan),
    }
    tuning_summary_rows = []
    tuning_fold_rows = []
    selection_rows = []
    fit_rows = []
    stability_rows = []
    fold_rows = []
    top_sets = {(route, int(fold)): set() for route in route_tags for fold in unique_folds}

    feature_names = (
        ["Stats120__" + format(i, "03d") for i in range(120)]
        + ["PairRaw465__" + format(i, "03d") for i in range(465)]
        + ["InterZ961__" + format(i, "03d") for i in range(961)]
    )

    for fold in unique_folds:
        train_index = np.flatnonzero(folds != fold)
        validation_index = np.flatnonzero(folds == fold)
        train_rows = [cohort_rows[i] for i in train_index]
        validation_rows = [cohort_rows[i] for i in validation_index]
        train_labels = labels[train_index]
        selected, tune_summary, tune_fold = tune_rssr(
            int(fold),
            train_rows,
            train_labels,
            structure[train_index],
            compact_by_sha,
            compact_schema,
            original_names,
            route_tags,
            args,
        )
        tuning_summary_rows.extend(tune_summary)
        tuning_fold_rows.extend(tune_fold)
        for route_tag in route_tags:
            selection_rows.append({
                "outer_fold": int(fold),
                "route": route_tag,
                **selected[route_tag],
            })

        x_train, x_validation, compact_names = audit.compact_matrix_variant(
            VARIANT,
            train_rows,
            validation_rows,
            compact_by_sha,
            compact_schema,
            original_names,
        )
        coarse, _, _ = audit.p72.fit_predict(
            x_train,
            train_labels,
            x_validation,
            seed=int(fold) - 1,
        )
        joint = audit.fit_config(
            x_train,
            x_validation,
            structure[train_index],
            structure[validation_index],
            train_labels,
            seed=int(fold) - 1,
        )
        (
            train_oof_probability,
            train_ambiguity,
            train_collision,
            train_novelty,
        ) = anchored.crossfit_base_and_risks(
            train_rows,
            train_labels,
            compact_by_sha,
            compact_schema,
            original_names,
            args.inner_folds,
            args.split_seed + 1000 * int(fold),
            args.split_seed + 10000 * int(fold),
            args.knn_k,
        )
        train_route_data = anchored.calibrated_routes(
            train_ambiguity,
            train_collision,
            train_novelty,
            train_ambiguity,
            train_collision,
            train_novelty,
            args.budget,
        )
        validation_collision, validation_novelty, _, _ = acn.neighborhood_risks(
            x_train, train_labels, x_validation, args.knn_k
        )
        validation_route_data = anchored.calibrated_routes(
            train_ambiguity,
            train_collision,
            train_novelty,
            acn.ambiguity(coarse),
            validation_collision,
            validation_novelty,
            args.budget,
        )
        standardized_train, standardized_validation, structure_audit = (
            audit.p72.standardize_added(
                structure[train_index],
                structure[validation_index],
                feature_names,
            )
        )
        usable = int(sum(
            not bool(item["constant_in_train"]) for item in structure_audit
        ))
        probability["C76"][validation_index] = coarse

        for route_tag, route_key in (
            (route_tags[0], "ambiguity_route"),
            (route_tags[1], "acn_route"),
        ):
            train_route = train_route_data[route_key]
            validation_route = validation_route_data[route_key]
            routes[route_tag][validation_index] = validation_route
            probability[route_tag + "__Joint"][validation_index] = acn.routed_probability(
                coarse, joint, validation_route
            )
            stability = stability_scores(
                standardized_train,
                train_labels,
                train_oof_probability,
                train_route,
                args.eta,
                args.stability_resamples,
                args.subsample_fraction,
                args.stability_seed
                + 1000000 * int(fold)
                + (0 if route_tag == route_tags[0] else 1),
                args.stability_epsilon,
            )
            weights = np.where(train_route, 1.0, float(args.eta))
            configurations = [
                ("Ordinary", 0.0, args.ordinary_l2),
                ("RSSR_fixed", args.fixed_gamma, args.fixed_l2),
                (
                    "RSSR_nested",
                    selected[route_tag]["gamma"],
                    selected[route_tag]["l2"],
                ),
            ]
            for model_tag, gamma, l2 in configurations:
                beta, diagnostic = fit_adaptive_residual(
                    standardized_train,
                    train_labels,
                    train_oof_probability,
                    weights,
                    stability["stability"],
                    stability["epsilon"],
                    gamma,
                    l2,
                    args.max_iterations,
                    args.optimizer_tolerance,
                )
                delta = standardized_validation @ beta
                enhanced = anchored.sigmoid(anchored.logit(coarse) + delta)
                method = route_tag + "__" + model_tag
                probability[method][validation_index] = acn.routed_probability(
                    coarse, enhanced, validation_route
                )
                fit_rows.append({
                    "outer_fold": int(fold),
                    "route": route_tag,
                    "model": model_tag,
                    "gamma": float(gamma),
                    "l2": float(l2),
                    "structure_dim": int(structure.shape[1]),
                    "usable_structure_dim": usable,
                    "train_routed_n": int(np.sum(train_route)),
                    "eval_delta_mean": float(np.mean(delta)),
                    "eval_delta_std": float(np.std(delta)),
                    "eval_delta_max_abs": float(np.max(np.abs(delta))),
                    **diagnostic,
                })

            stability_value = stability["stability"]
            q25, q75 = np.quantile(stability_value, [0.25, 0.75])
            top_count = max(1, int(round(0.25 * len(stability_value))))
            top_index = np.argsort(-stability_value, kind="mergesort")[:top_count]
            top_sets[(route_tag, int(fold))] = set(map(int, top_index))
            for feature_index, feature_name in enumerate(feature_names):
                value = float(stability_value[feature_index])
                group = "Medium"
                if value <= q25:
                    group = "Unstable"
                elif value >= q75:
                    group = "Stable"
                stability_rows.append({
                    "outer_fold": int(fold),
                    "route": route_tag,
                    "feature_index": int(feature_index),
                    "feature_name": feature_name,
                    "feature_family": feature_name.split("__", 1)[0],
                    "consistency": float(stability["consistency"][feature_index]),
                    "strength": float(stability["strength"][feature_index]),
                    "mad": float(stability["mad"][feature_index]),
                    "stability": value,
                    "stability_group": group,
                    "top_quartile": int(feature_index in top_sets[(route_tag, int(fold))]),
                    "constant_in_outer_train": int(
                        bool(structure_audit[feature_index]["constant_in_train"])
                    ),
                })
        for key in risk_outputs:
            risk_outputs[key][validation_index] = validation_route_data[key]
        fold_rows.append({
            "outer_fold": int(fold),
            "outer_train_n": int(len(train_index)),
            "outer_validation_n": int(len(validation_index)),
            "compact_dim": int(len(compact_names)),
            "structure_dim": int(structure.shape[1]),
            "usable_structure_dim": usable,
        })
        print("outer fold", int(fold), "done", flush=True)

    if any(np.any(~np.isfinite(value)) for value in probability.values()):
        raise RuntimeError("incomplete OOF probability")
    if any(np.any(~np.isfinite(value)) for value in risk_outputs.values()):
        raise RuntimeError("incomplete OOF risk")

    zero_route = np.zeros(n, dtype=bool)
    summary_rows = [
        pairz_exp.metric_row(
            "C76", labels, probability["C76"], probability["C76"], zero_route
        )
    ]
    per_fold_rows = []
    for fold in unique_folds:
        selected_index = folds == fold
        per_fold_rows.append({
            **pairz_exp.metric_row(
                "C76",
                labels[selected_index],
                probability["C76"][selected_index],
                probability["C76"][selected_index],
                zero_route[selected_index],
            ),
            "fold": int(fold),
        })
    for method in methods[1:]:
        route_tag = method.split("__", 1)[0]
        route = routes[route_tag]
        summary_rows.append(pairz_exp.metric_row(
            method, labels, probability[method], probability["C76"], route
        ))
        for fold in unique_folds:
            selected_index = folds == fold
            per_fold_rows.append({
                **pairz_exp.metric_row(
                    method,
                    labels[selected_index],
                    probability[method][selected_index],
                    probability["C76"][selected_index],
                    route[selected_index],
                ),
                "fold": int(fold),
            })

    comparison_pairs = []
    for route_tag in route_tags:
        comparison_pairs.extend([
            (route_tag + "__Joint", route_tag + "__Ordinary"),
            (route_tag + "__Ordinary", route_tag + "__RSSR_fixed"),
            (route_tag + "__Ordinary", route_tag + "__RSSR_nested"),
            (route_tag + "__RSSR_fixed", route_tag + "__RSSR_nested"),
        ])
    for model_tag in model_tags:
        comparison_pairs.append((
            route_tags[0] + "__" + model_tag,
            route_tags[1] + "__" + model_tag,
        ))
    pairwise_rows = []
    for left, right in comparison_pairs:
        left_correct = (probability[left] >= THRESHOLD) == labels
        right_correct = (probability[right] >= THRESHOLD) == labels
        left_only, right_only, p_value = pairz_exp.exact_mcnemar(
            left_correct, right_correct
        )
        pairwise_rows.append({
            "left": left,
            "right": right,
            "left_only_correct": int(left_only),
            "right_only_correct": int(right_only),
            "net_right_minus_left": int(right_only - left_only),
            "discordant": int(left_only + right_only),
            "mcnemar_exact_p": float(p_value),
        })

    overlap_rows = []
    for route_tag in route_tags:
        for i, left_fold in enumerate(unique_folds):
            for right_fold in unique_folds[i + 1:]:
                left = top_sets[(route_tag, int(left_fold))]
                right = top_sets[(route_tag, int(right_fold))]
                union = left | right
                overlap_rows.append({
                    "route": route_tag,
                    "left_fold": int(left_fold),
                    "right_fold": int(right_fold),
                    "intersection": int(len(left & right)),
                    "union": int(len(union)),
                    "jaccard": float(len(left & right) / len(union)) if union else 1.0,
                })

    prediction_rows = []
    for index, row in enumerate(manifest):
        output = {
            "sha256": row["sha256"],
            "label": int(labels[index]),
            "fold": int(folds[index]),
            "ambiguity_routed": int(routes[route_tags[0]][index]),
            "acn_routed": int(routes[route_tags[1]][index]),
            **{key: float(value[index]) for key, value in risk_outputs.items()},
        }
        for method in methods:
            output[method + "__prob"] = float(probability[method][index])
        prediction_rows.append(output)

    reference = check_previous_predictions(
        args.previous_predictions,
        manifest,
        probability,
        routes,
        route_tags,
    )
    write_csv(args.output / "source_summary.csv", summary_rows)
    write_csv(args.output / "source_per_fold.csv", per_fold_rows)
    write_csv(args.output / "source_predictions.csv", prediction_rows)
    write_csv(args.output / "pairwise_tests.csv", pairwise_rows)
    write_csv(args.output / "fit_audit.csv", fit_rows)
    write_csv(args.output / "tuning_summary.csv", tuning_summary_rows)
    write_csv(args.output / "tuning_fold_scores.csv", tuning_fold_rows)
    write_csv(args.output / "outer_selection.csv", selection_rows)
    write_csv(args.output / "stability_feature_audit.csv", stability_rows)
    write_csv(args.output / "stability_top_quartile_overlap.csv", overlap_rows)
    write_csv(args.output / "fold_audit.csv", fold_rows)
    (args.output / "reference_check.json").write_text(
        json.dumps({
            "multiscale_features": structure_reference,
            "previous_predictions": reference,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metadata = {
        "experiment": "rssr-interz-2017-v1",
        "source_only": True,
        "source_apks": int(n),
        "future_rows_used_for_model_or_selection": False,
        "future_labels_used": False,
        "input_feature_table_note": (
            "load_inputs scans features.csv to index rows by SHA, then restricts all matrices "
            "to the fixed 799-APK 2017 manifest"
        ),
        "outer_folds": [int(value) for value in unique_folds],
        "structure": STRUCTURE_TAG,
        "structure_dim": int(structure.shape[1]),
        "routes": route_tags,
        "budget": float(args.budget),
        "eta": float(args.eta),
        "stability": {
            "resamples": int(args.stability_resamples),
            "subsample_fraction": float(args.subsample_fraction),
            "stratified": True,
            "epsilon": float(args.stability_epsilon),
            "hard_feature_deletion": False,
        },
        "fixed_rssr": {
            "gamma": float(args.fixed_gamma),
            "l2": float(args.fixed_l2),
        },
        "nested_tuning": {
            "gamma_grid": [float(value) for value in args.gamma_grid],
            "l2_grid": [float(value) for value in args.l2_grid],
            "tuning_folds": int(args.tuning_folds),
            "base_crossfit_folds_inside_tuning_train": int(args.tuning_crossfit_folds),
            "selection_metric": "mean routed balanced accuracy; tie-break min BA, lower gamma, L2 closest to 0.03",
        },
        "strict_nesting": (
            "standardization, zero-variance handling, C76 train OOF probability, routing, "
            "stability scores, and RSSR selection are recomputed within each applicable train split"
        ),
        "decision_rule": (
            "Retain RSSR only if it reduces breaks or improves worst-fold/stability relative to "
            "ordinary L2 without sacrificing global BA. Otherwise retain Joint or ordinary residual."
        ),
        "reference_reproduction": reference,
    }
    (args.output / "audit.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Done:", args.output)


if __name__ == "__main__":
    main()
